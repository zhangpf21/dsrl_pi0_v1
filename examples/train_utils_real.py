import os
import time
from tqdm import tqdm
import time
import numpy as np
import jax
import sys
import select
import tty
import termios
from openpi_client import image_tools
from moviepy.editor import ImageSequenceClip


def _adapter_conditioning_enabled(variant):
    return bool(getattr(variant, 'train_kwargs', {}).get('use_adapter_conditioning', 0))


def _adapter_dims(variant):
    train_kwargs = getattr(variant, 'train_kwargs', {})
    return (
        int(train_kwargs.get('noise_dim', 32)),
        int(train_kwargs.get('control_dim', 16)),
        int(train_kwargs.get('adapter_feature_dim', 1024)),
        int(train_kwargs.get('adapter_gate_dim', 1)),
    )


def _repeat_to_horizon(x, horizon):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] >= horizon:
        return x[:horizon]
    tail = np.repeat(x[-1:, :], horizon - x.shape[0], axis=0)
    return np.concatenate([x, tail], axis=0)


def _split_adapter_action(packed_action, variant):
    noise_dim, _, adapter_feature_dim, adapter_gate_dim = _adapter_dims(variant)
    noise = packed_action[..., :noise_dim]
    adapter_start = noise_dim
    adapter_end = adapter_start + adapter_feature_dim
    gate_end = adapter_end + adapter_gate_dim
    return noise, packed_action[..., adapter_start:adapter_end], packed_action[..., adapter_end:gate_end]


def _make_initial_action_and_controls(key, agent, variant, horizon):
    if not _adapter_conditioning_enabled(variant):
        noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
        noise_repeat = jax.numpy.repeat(noise[:, -1:, :], horizon - noise.shape[1], axis=1)
        noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
        return np.asarray(noise[0, :agent.action_chunk_shape[0], :]), noise, None, None

    noise_dim, control_dim, adapter_feature_dim, adapter_gate_dim = _adapter_dims(variant)
    chunk_len = agent.action_chunk_shape[0]
    noise = np.asarray(jax.random.normal(key, (chunk_len, noise_dim)), dtype=np.float32)
    control_code = np.zeros((chunk_len, control_dim), dtype=np.float32)
    gate_code = -np.ones((chunk_len, adapter_gate_dim), dtype=np.float32)
    adapter_feature = np.zeros((horizon, adapter_feature_dim), dtype=np.float32)
    adapter_gate = np.zeros((horizon, adapter_gate_dim), dtype=np.float32)
    latent_action = np.concatenate([noise, control_code, gate_code], axis=-1)
    return (
        latent_action,
        _repeat_to_horizon(noise, horizon)[None],
        adapter_feature[None],
        adapter_gate[None],
    )


def _make_controls_from_actor_action(packed_action, variant, horizon):
    packed_action = np.asarray(packed_action, dtype=np.float32)
    if not _adapter_conditioning_enabled(variant):
        noise = _repeat_to_horizon(packed_action, horizon)[None]
        return noise, None, None

    noise, adapter_feature, adapter_gate = _split_adapter_action(packed_action, variant)
    return (
        _repeat_to_horizon(noise, horizon)[None],
        _repeat_to_horizon(adapter_feature, horizon)[None],
        _repeat_to_horizon(adapter_gate, horizon)[None],
    )


def _infer_pi0(agent_dp, request_data, noise, adapter_feature=None, adapter_gate=None):
    kwargs = {}
    if adapter_feature is not None:
        kwargs['adapter_feature'] = np.asarray(adapter_feature, dtype=np.float32)
    if adapter_gate is not None:
        kwargs['adapter_gate'] = np.asarray(adapter_gate, dtype=np.float32)
    return agent_dp.infer(request_data, noise=np.asarray(noise, dtype=np.float32), **kwargs)


def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       shard_fn=None, agent_dp=None, robot_config=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)
        
    i = 0
    total_env_steps = 0
    total_num_traj = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
   
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            traj = collect_traj(variant, agent, env, i, agent_dp, wandb_logger, total_num_traj, robot_config)
            total_num_traj += 1
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', total_num_traj)
            print('total env steps:', total_env_steps)
            
            if i == 0:
                num_gradsteps = 5000
            else:
                num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step
            print(f'num_gradsteps: {num_gradsteps}')
            if total_num_traj >= variant.num_initial_traj_collect:
                for _ in range(num_gradsteps):

                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1
                    
                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': total_num_traj}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1:
                        if i % variant.checkpoint_interval == 0:
                            agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
            
def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    
    discount_horizon = variant.query_freq
    actions = np.array(traj['actions']) # (T, chunk_size, DSRL action dim)
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        
        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()

def collect_traj(variant, agent, env, i, agent_dp=None, wandb_logger=None, traj_id=None, robot_config=None):
    query_frequency = variant.query_freq
    instruction = variant.instruction
    max_timesteps = robot_config['max_timesteps']
    agent._rng, rng = jax.random.split(agent._rng)
    try:
        env.reset()
    except Exception as e:
        print(f"Environment reset failed")
        import traceback
        traceback.print_exc() 
        import pdb; pdb.set_trace()
    step_time = 1 / 15 # 15 Hz
    last_step_time = time.time()
    old_settings = termios.tcgetattr(sys.stdin)
    
    rewards = []
    action_list = []
    obs_list = []
    image_list = []

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        for t in tqdm(range(max_timesteps)):    
            # Check for keyboard input
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input.lower() == 'q':
                    print("'q' pressed, stopping loop.")
                    break
            
            try:
                _env_obs = env.get_observation()
            except Exception as e:
                print(f"Environment get obs failed")
                import traceback
                traceback.print_exc()
                import pdb; pdb.set_trace()
            curr_obs = _extract_observation(
                    robot_config,
                    _env_obs,
            )
            image_list.append(curr_obs[robot_config['camera_to_use'] + "_image"])

            request_data = get_pi0_input(curr_obs, robot_config, instruction)
        
            if t % query_frequency == 0:

                rng, key = jax.random.split(rng)

                img_all = process_images(variant, curr_obs)
                
                # extract the feature from the pi0 VLM backbone and concat with the qpos as states
                img_rep_pi0, _ = agent_dp.get_prefix_rep(request_data)
                img_rep_pi0 = img_rep_pi0[:, -1, :] # (1, 2048)
                qpos = np.concatenate([curr_obs["joint_position"], curr_obs["gripper_position"], img_rep_pi0.flatten()])

                obs_dict = {
                    'pixels': img_all,
                    'state': qpos[np.newaxis, ..., np.newaxis],
                }
                if i == 0:
                    actions_noise, noise, adapter_feature, adapter_gate = _make_initial_action_and_controls(
                        key, agent, variant, horizon=10
                    )
                else:
                    # SAC predicts the low-dimensional latent action; pack it only for pi0 execution.
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    packed_action = agent.pack_actions(obs_dict, actions_noise)
                    packed_action = np.reshape(packed_action, (agent.action_chunk_shape[0], -1))
                    noise, adapter_feature, adapter_gate = _make_controls_from_actor_action(
                        packed_action, variant, horizon=10
                    )
                action_list.append(actions_noise)
                obs_list.append(obs_dict)
                action = _infer_pi0(
                    agent_dp,
                    request_data,
                    noise=noise,
                    adapter_feature=adapter_feature,
                    adapter_gate=adapter_gate,
                )["actions"]

            action_t = action[t % query_frequency]
            
            # binarize gripper action.
            if action_t[-1].item() > 0.5:
                action_t = np.concatenate([action_t[:-1], np.ones((1,))])
            else:
                action_t = np.concatenate([action_t[:-1], np.zeros((1,))])
            action_t = np.clip(action_t, -1, 1)
            
            try:
                env.step(action_t)
            except Exception as e:
                print(f"Environment step failed")
                import traceback
                traceback.print_exc()  # This prints the full traceback
                import pdb; pdb.set_trace()
        
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now
            
        print("Trial finished. Mark as (1) Success or (0) Failure:")
        while True:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input == '1':
                    print("Trial marked as SUCCESS.")
                    is_success = True
                    break
                elif char_input == '0':
                    print("Trial marked as FAILURE.")                    
                    is_success = False
                    break
                else:
                    print("Invalid input. Please enter '1' for Success or '0' for Failure:")
            time.sleep(0.01) # Small sleep to prevent busy-waiting if no input

        try:
            _env_obs = env.get_observation()
        except Exception as e:
            print(f"Environment get obs failed")
            import traceback
            traceback.print_exc()
            import pdb; pdb.set_trace()
        
        # add last observation
        curr_obs = _extract_observation(
                    robot_config,
                    _env_obs,
            )
        image_list.append(curr_obs[robot_config['camera_to_use'] + "_image"])
        request_data = get_pi0_input(curr_obs, robot_config, instruction)
        img_all = process_images(variant, curr_obs)
        img_rep_pi0, _ = agent_dp.get_prefix_rep(request_data)
        img_rep_pi0 = img_rep_pi0[:, -1, :] # (1, 2048)
        qpos = np.concatenate([curr_obs["joint_position"], curr_obs["gripper_position"], img_rep_pi0.flatten()])
        obs_dict = {
            'pixels': img_all,
            'state': qpos[np.newaxis, ..., np.newaxis],
        }
        obs_list.append(obs_dict)
        print(f'Rollout Done')
        
    finally:
        if is_success:
            query_steps = len(action_list)
            rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            query_steps = len(action_list)
            rewards = -np.ones(query_steps)
            masks = np.ones(query_steps)
            
        if wandb_logger is not None:
            wandb_logger.log({f'is_success': int(is_success)}, step=i)
            wandb_logger.log({f'total_num_traj': traj_id}, step=i)

        video_path = os.path.join(variant.outputdir, f'video_high_{traj_id}.mp4')
        video = np.stack(image_list)
        ImageSequenceClip(list(video), fps=15).write_videofile(video_path, codec="libx264")
       
        print("Episide Done! Press c after resetting the environment")
        try:
            env.reset()
        except Exception as e:
            print(f"Environment reset failed")
            import traceback
            traceback.print_exc()  # This prints the full traceback
            import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    traj = {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'env_steps': t + 1,
    }
    
    return traj


def _extract_observation(robot_config, obs_dict):
    '''
    from https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py
    '''
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations.keys():
        if robot_config['left_camera_id'] in key and "left" in key:
            left_image = image_observations[key]
        elif robot_config['right_camera_id'] in key and "left" in key:
            right_image = image_observations[key]
        elif robot_config['wrist_camera_id'] in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }
    
def get_pi0_input(obs, robot_config, instruction):
    external_image = obs[robot_config['camera_to_use'] + "_image"]
    request_data = {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            external_image, 224, 224
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(obs["wrist_image"], 224, 224),
        "observation/joint_position": obs["joint_position"],
        "observation/gripper_position": obs["gripper_position"],
        "prompt": instruction,
    }
    return request_data
    

def process_images(variant, obs):
    '''
    concat the images from all cameras
    '''
    im1 = image_tools.resize_with_pad(obs["left_image"], variant.resize_image, variant.resize_image)
    im2 = image_tools.resize_with_pad(obs["right_image"], variant.resize_image, variant.resize_image)
    im3 = image_tools.resize_with_pad(obs["wrist_image"], variant.resize_image, variant.resize_image)
    img_all = np.concatenate([im1, im2, im3], axis=2)[np.newaxis, ..., np.newaxis]
    return img_all
