from typing import Callable, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init

class LearnedStdNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return distribution

class TanhMultivariateNormalDiag(distrax.Transformed):

    def __init__(self,
                 loc: jnp.ndarray,
                 scale_diag: jnp.ndarray,
                 low: Optional[jnp.ndarray] = None,
                 high: Optional[jnp.ndarray] = None):
        distribution = distrax.MultivariateNormalDiag(loc=loc,
                                                      scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1))

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

class LearnedStdTanhNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    low: Optional[float] = None
    high: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = TanhMultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds), low=self.low, high=self.high)
        return distribution


class AdapterConditionedDistribution:
    """Distribution over low-dim controls that returns packed adapter actions.

    The stochastic policy samples [noise, control_code, gate_raw], then a
    trainable adapter head deterministically maps control_code to the high-dim
    adapter feature consumed by pi0.
    """

    def __init__(
        self,
        latent_distribution: distrax.Distribution,
        adapter_fn: Callable[[jnp.ndarray], jnp.ndarray],
        noise_dim: int,
        control_dim: int,
        adapter_feature_dim: int,
        gate_dim: int,
        noise_scale: float = 1.0,
        gate_logit_scale: float = 5.0,
        gate_logit_bias: float = -2.5,
    ):
        self.latent_distribution = latent_distribution
        self.distribution = latent_distribution.distribution
        self.noise_dim = noise_dim
        self.control_dim = control_dim
        self.adapter_feature_dim = adapter_feature_dim
        self.gate_dim = gate_dim
        self.noise_scale = noise_scale
        self.gate_logit_scale = gate_logit_scale
        self.gate_logit_bias = gate_logit_bias
        self._adapter_fn = adapter_fn

    def _pack(self, latent_action: jnp.ndarray) -> jnp.ndarray:
        noise_end = self.noise_dim
        control_end = noise_end + self.control_dim
        gate_end = control_end + self.gate_dim

        noise = latent_action[..., :noise_end] * self.noise_scale
        control_code = latent_action[..., noise_end:control_end]
        gate_code = latent_action[..., control_end:gate_end]
        gate_logits = self.gate_logit_scale * gate_code + self.gate_logit_bias
        gate = 1.0 / (1.0 + jnp.exp(-gate_logits))
        adapter_feature = self._adapter_fn(control_code)
        return jnp.concatenate([noise, adapter_feature, gate], axis=-1)

    def sample(self, seed: jnp.ndarray, sample_shape=()) -> jnp.ndarray:
        return self._pack(self.latent_distribution.sample(seed=seed, sample_shape=sample_shape))

    def sample_and_log_prob(self, seed: jnp.ndarray, sample_shape=()):
        latent_action, log_prob = self.latent_distribution.sample_and_log_prob(seed=seed, sample_shape=sample_shape)
        return self._pack(latent_action), log_prob

    def mode(self) -> jnp.ndarray:
        return self._pack(self.latent_distribution.mode())

    @property
    def loc(self) -> jnp.ndarray:
        return self.mode()

    def log_prob(self, actions: jnp.ndarray) -> jnp.ndarray:
        # This is only used by auxiliary eval helpers. The adapter map is
        # many-to-one, so we cannot invert adapter_feature back to control_code.
        # Use zero control_code and the stored gate for a coarse diagnostic logp.
        if actions.ndim > 2:
            actions = jnp.reshape(actions, (*actions.shape[:-2], actions.shape[-2] * actions.shape[-1]))
        noise = actions[..., :self.noise_dim] / self.noise_scale
        gate = actions[..., self.noise_dim + self.adapter_feature_dim : self.noise_dim + self.adapter_feature_dim + self.gate_dim]
        gate = jnp.clip(gate, 1e-6, 1.0 - 1e-6)
        gate_logits = jnp.log(gate) - jnp.log1p(-gate)
        gate_code = (gate_logits - self.gate_logit_bias) / self.gate_logit_scale
        gate_code = jnp.clip(gate_code, -1.0 + 1e-6, 1.0 - 1e-6)
        control_code = jnp.zeros((*noise.shape[:-1], self.control_dim), dtype=actions.dtype)
        latent_action = jnp.concatenate([noise, control_code, gate_code], axis=-1)
        return self.latent_distribution.log_prob(latent_action)


class AdapterConditionedTanhNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    noise_dim: int = 32
    control_dim: int = 16
    adapter_feature_dim: int = 1024
    gate_dim: int = 1
    adapter_hidden_dim: int = 128
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    noise_scale: float = 1.0
    gate_logit_scale: float = 5.0
    gate_logit_bias: float = -2.5

    @nn.compact
    def __call__(
            self,
            observations: jnp.ndarray,
            training: bool = False) -> AdapterConditionedDistribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        latent_dim = self.noise_dim + self.control_dim + self.gate_dim
        means = nn.Dense(latent_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = nn.Dense(latent_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        latent_distribution = TanhMultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds),
        )

        adapter_w1 = self.param(
            "adapter_fc1_kernel",
            default_init(),
            (self.control_dim, self.adapter_hidden_dim),
        )
        adapter_b1 = self.param(
            "adapter_fc1_bias",
            nn.initializers.zeros,
            (self.adapter_hidden_dim,),
        )
        adapter_w2 = self.param(
            "adapter_fc2_kernel",
            nn.initializers.zeros,
            (self.adapter_hidden_dim, self.adapter_feature_dim),
        )
        adapter_b2 = self.param(
            "adapter_fc2_bias",
            nn.initializers.zeros,
            (self.adapter_feature_dim,),
        )

        def adapter_fn(control_code: jnp.ndarray) -> jnp.ndarray:
            x = jnp.matmul(control_code, adapter_w1) + adapter_b1
            x = nn.relu(x)
            return jnp.tanh(jnp.matmul(x, adapter_w2) + adapter_b2)

        return AdapterConditionedDistribution(
            latent_distribution,
            adapter_fn,
            noise_dim=self.noise_dim,
            control_dim=self.control_dim,
            adapter_feature_dim=self.adapter_feature_dim,
            gate_dim=self.gate_dim,
            noise_scale=self.noise_scale,
            gate_logit_scale=self.gate_logit_scale,
            gate_logit_bias=self.gate_logit_bias,
        )
