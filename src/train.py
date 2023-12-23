import torch
import os
import numpy as np
import gym
import utils
import time
import wandb
from arguments import parse_args
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder, MaskRecorder, AugmentationRecorder


def record_video(env, agent, video, step, test_env=False, test_mode=None):
    obs = env.reset()
    video.init()
    done = False
    with torch.no_grad():
        with utils.eval_mode(agent):
            while not done:
                action = agent.select_action(obs, test_env)
                obs, _, done, _ = env.step(action)
                video.record(env)
    _test_env = f'_test_env_{test_mode}' if test_env else ''
    video.save(f'{step}{_test_env}.mp4')


def evaluate(env, agent, mask_rec, args, L, step, test_env=False, test_mode=None):
    episode_rewards = []
    for i in range(args.eval_episodes):
        obs = env.reset()
        mask_rec.init(enabled=(i == 0))
        done = False
        episode_reward, episode_step = 0, 0
        # torch_obs, torch_action = [], []
        with torch.no_grad():
            with utils.eval_mode(agent):
                while not done:
                    action = agent.select_action(obs, test_env)
                    obs, reward, done, _ = env.step(action)
                    mask_rec.record(obs, agent, episode_step, step, test_env, test_mode, L)
                    # if args.algorithm == 'sgqn':
                    #     utils.logging_sgqn(agent, i, episode_step, step, obs, torch_obs, torch_action, action, test_mode)
                    episode_reward += reward
                    episode_step += 1

        if L is not None:
            _test_env = f'_test_env_{test_mode}' if test_env else ''
            L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)


def main(args):
    # Init wandb
    wandb.init(project="madi", config=vars(args), name=utils.set_experiment_name(args),
               entity="bramgrooten", mode=args.wandb_mode)
    start_training_time = time.time()
    utils.set_seed_everywhere(args.seed)

    # Initialize environments
    gym.logger.set_level(40)
    env, train_env_eval, test_envs, test_modes = utils.make_envs(args)

    # Create working directory
    work_dir = os.path.join(args.log_dir, args.domain_name + '_' + args.task_name, args.algorithm + '_' + args.encoder, str(args.seed))
    print('Working directory:', work_dir)
    # assert not os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
    utils.make_dir(work_dir)
    model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
    mask_dir = utils.make_dir(os.path.join(work_dir, 'mask'))
    aug_dir = utils.make_dir(os.path.join(work_dir, 'aug'))
    mask_rec = MaskRecorder(mask_dir if args.save_mask else None, args)
    aug_rec = AugmentationRecorder(aug_dir, args.save_aug, args)
    agent_args = {'aug_recorder': aug_rec} if args.algorithm == 'madi' else {}
    utils.write_info(args, os.path.join(work_dir, 'info.log'))

    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.train_steps,
        batch_size=args.batch_size
    )
    cropped_obs_shape = (3 * args.frame_stack, args.image_crop_size, args.image_crop_size)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=env.action_space.shape,
        args=args,
        agent_args=agent_args
    )

    start_step, episode, episode_reward, done = 0, 0, 0, True
    L = Logger(work_dir)
    start_time = time.time()
    for step in range(start_step, args.train_steps + 1):
        if done:
            if step > start_step:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # Evaluate agent periodically
            if step % args.eval_freq == 0:
                print('Evaluating:', work_dir)
                L.log('eval/episode', episode, step)
                evaluate(train_env_eval, agent, mask_rec, args, L, step)
                if test_envs is not None:
                    for test_env, test_mode in zip(test_envs, test_modes):
                        evaluate(test_env, agent, mask_rec, args, L, step, test_env=True, test_mode=test_mode)
                L.dump(step)

            # Periodically save the agent
            if (step > start_step and step % args.save_freq == 0) or step == 1e5:
                torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

            # Periodically record a video
            if step > start_step and step % args.save_vid_every == 0:
                record_video(train_env_eval, agent, video, step)
                if test_envs is not None:
                    for test_env, test_mode in zip(test_envs, test_modes):
                        record_video(test_env, agent, video, step, test_env=True, test_mode=test_mode)

            L.log('train/episode_reward', episode_reward, step)
            L.log('train/episode', episode + 1, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

        # Sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training updates for initial random steps
        if step == args.init_steps:
            for i in range(1, args.init_steps + 1):
                agent.update(replay_buffer, L, i)

        # Run training update
        if step > args.init_steps:
            agent.update(replay_buffer, L, step)

        # Take step
        next_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        episode_step += 1
        obs = next_obs

    L.log('train/total_training_hours', (time.time() - start_training_time) / 3600., step=args.train_steps)
    print('Completed training for', work_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
