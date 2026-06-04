from audioop import cross
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey


def update_actor(key: PRNGKey, actor: TrainState, critic: TrainState,
                 temp: TrainState, batch: DatasetDict, cross_norm:bool=False, critic_reduction:str='min',
                 use_adapter_conditioning: bool = False, noise_dim: int = 32,
                 adapter_feature_dim: int = 1024, adapter_gate_dim: int = 1,
                 adapter_l2_coef: float = 1e-4, gate_l1_coef: float = 1e-4) -> Tuple[TrainState, Dict[str, float]]:
    
    key, key_act = jax.random.split(key, num=2)

    def actor_loss_fn(
            actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['observations'], mutable=['batch_stats'])
            if cross_norm:
                next_dist = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['next_observations'], mutable=['batch_stats'])
            else:
                next_dist = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['next_observations'])
            if type(next_dist) == tuple:
                next_dist, new_model_state = next_dist
        else:
            dist = actor.apply_fn({'params': actor_params}, batch['observations'])
            next_dist = actor.apply_fn({'params': actor_params}, batch['next_observations'])
            new_model_state = {}
        
        # For logging only
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        mean_dist_norm = jnp.linalg.norm(mean_dist, axis=-1)
        std_dist_norm = jnp.linalg.norm(std_diag_dist, axis=-1)

        
        actions, log_probs = dist.sample_and_log_prob(seed=key_act)

        if hasattr(critic, 'batch_stats') and critic.batch_stats is not None:
            qs, _ = critic.apply_fn({'params': critic.params, 'batch_stats': critic.batch_stats}, batch['observations'],
                            actions, mutable=['batch_stats'])
        else:    
            qs = critic.apply_fn({'params': critic.params}, batch['observations'], actions)
        
        if critic_reduction == 'min':
            q = qs.min(axis=0)
        elif critic_reduction == 'mean':
            q = qs.mean(axis=0)
        else:
            raise ValueError(f"Invalid critic reduction: {critic_reduction}")
        actor_loss = (log_probs * temp.apply_fn({'params': temp.params}) - q).mean()
        adapter_l2 = jnp.array(0.0)
        gate_l1 = jnp.array(0.0)
        adapter_feature_norm = jnp.array(0.0)
        adapter_gate_mean = jnp.array(0.0)
        if use_adapter_conditioning:
            adapter_start = noise_dim
            adapter_end = adapter_start + adapter_feature_dim
            gate_end = adapter_end + adapter_gate_dim
            packed_actions = dist.pack(actions)
            adapter_feature = packed_actions[..., adapter_start:adapter_end]
            adapter_gate = packed_actions[..., adapter_end:gate_end]
            adapter_l2 = jnp.mean(jnp.square(adapter_feature))
            gate_l1 = jnp.mean(jnp.abs(adapter_gate))
            adapter_feature_norm = jnp.linalg.norm(adapter_feature, axis=-1).mean()
            adapter_gate_mean = adapter_gate.mean()
            actor_loss = actor_loss + adapter_l2_coef * adapter_l2 + gate_l1_coef * gate_l1

        things_to_log = {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'q_pi_in_actor': q.mean(),
            'adapter_l2': adapter_l2,
            'adapter_gate_l1': gate_l1,
            'adapter_feature_norm': adapter_feature_norm,
            'adapter_gate_mean': adapter_gate_mean,
            'mean_pi_norm': mean_dist_norm.mean(),
            'std_pi_norm': std_dist_norm.mean(),
            'mean_pi_avg': mean_dist.mean(),
            'mean_pi_max': mean_dist.max(),
            'mean_pi_min': mean_dist.min(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_max': std_diag_dist.max(),
            'std_pi_min': std_diag_dist.min(),
        }
        return actor_loss, (things_to_log, new_model_state)

    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info
