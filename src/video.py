import os
import imageio
import torch
import torchvision
from numpy import ndarray
from argparse import Namespace
from logger import Logger


class VideoRecorder(object):
    def __init__(self, dir_name, height=448, width=448, camera_id=0, fps=25):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env, mode=None):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            if mode is not None and 'video' in mode:
                _env = env
                while 'video' not in _env.__class__.__name__.lower():
                    _env = _env.env
                frame = _env.apply_to(frame)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)


class MaskRecorder(object):
    def __init__(self, dir_name, args):
        self.dir_name = dir_name
        self.save_episode_step_num = 10
        self.algorithm = args.algorithm
        self.grey_transform = torchvision.transforms.ToPILImage(mode='L')
        self.num_frames = args.frame_stack
        self.num_masks = args.frame_stack
        self.mask_type = args.mask_type
        self.save_all_frames = False  # set to True to save all (3) frames, instead of just the first one

    def init(self, enabled=True):
        self.obses = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, obs, agent, episode_step, training_step, test_env, test_mode, logger=None):
        if self.enabled and episode_step == self.save_episode_step_num:
            if training_step <= 100_000 or training_step % 100_000 == 0:
                _test_env_name = f'_test_{test_mode}' if test_env else ''
                obs = agent._obs_to_input(obs)
                self.save_obs_per_frame(obs, training_step, _test_env_name)
                if self.algorithm == 'madi':
                    self.save_masked_obs_per_frame(obs, agent, training_step, test_env, _test_env_name)
                    self.save_mask_per_frame(obs, agent, training_step, test_env, _test_env_name, logger)
                    if self.mask_type == 'hard' or self.mask_type == 'mixed' and test_env:
                        # also save the soft mask in this case (for comparison)
                        self.save_soft_mask(obs, agent, training_step, _test_env_name, logger)

    def save_obs_per_frame(self, obs, training_step, _test_env_name):
        for frame in range(self.num_frames):
            if frame == 0 or self.save_all_frames:
                torchvision.utils.save_image(
                    obs[0][3 * frame:3 * frame + 3] / 255.,
                    os.path.join(self.dir_name, f'step{training_step}{_test_env_name}_frame{frame}_obs.png'))

    def save_masked_obs_per_frame(self, obs, agent, training_step, test_env, _test_env_name):
        masked_obs = agent.apply_mask(obs, test_env)
        for frame in range(self.num_frames):
            if frame == 0 or self.save_all_frames:
                torchvision.utils.save_image(
                    masked_obs[0][3 * frame:3 * frame + 3] / 255.,
                    os.path.join(self.dir_name, f'step{training_step}{_test_env_name}_frame{frame}_masked_obs.png'))

    def save_mask_per_frame(self, obs, agent, training_step, test_env, _test_env_name, logger=None):
        # per frame only if we perform frame-wise masking
        if self.num_masks == 1:
            mask = agent.masker(obs, test_env=test_env)
            mask_image = self.grey_transform(mask.squeeze())
            mask_image.save(os.path.join(self.dir_name, f'step{training_step}{_test_env_name}_mask.png'))
        else:
            frames = obs.chunk(self.num_frames, dim=1)
            for f in range(self.num_frames):
                if f == 0 or self.save_all_frames:
                    mask = agent.masker(frames[f], test_env=test_env)
                    mask_image = self.grey_transform(mask.squeeze())
                    mask_image.save(os.path.join(self.dir_name, f'step{training_step}{_test_env_name}_frame{f}_mask.png'))
        if self.mask_type == 'soft':
            log_soft_mask_stats(agent, mask, logger, training_step, _test_env_name)

    def save_soft_mask(self, obs, agent, training_step, _test_env_name, logger=None):
        # per frame only if we perform frame-wise masking
        if self.num_masks == 1:
            agent.masker(obs, log_mask_stats=True)
            soft_mask = agent.masker.current_soft_mask
            mask_image = self.grey_transform(soft_mask.squeeze())
            mask_image.save(os.path.join(self.dir_name, f'step{training_step}{_test_env_name}_soft_mask.png'))
        else:
            frames = obs.chunk(self.num_frames, dim=1)
            for f in range(self.num_frames):
                if f == 0 or self.save_all_frames:
                    agent.masker(frames[f], log_mask_stats=True)
                    soft_mask = agent.masker.current_soft_mask
                    mask_image = self.grey_transform(soft_mask.squeeze())
                    mask_image.save(os.path.join(self.dir_name, f'step{training_step}{_test_env_name}_frame{f}_soft_mask.png'))
        if logger is not None:
            log_soft_mask_stats(agent, soft_mask, logger, training_step, _test_env_name)


class AugmentationRecorder(object):
    def __init__(self, dir_name: str, enabled: bool, args: Namespace):
        self.dir_name = dir_name
        self.enabled = enabled
        self.save_every = args.save_aug_every
        self.num_frames = args.frame_stack
        self.save_all_frames = args.save_all_frames
        self.extra_record_steps = [10, 50, 100, 500, 2000, 5000]

    def record(self, obs: ndarray, step: int, masked: bool = False) -> None:
        if self.enabled and (step % self.save_every == 0 or step in self.extra_record_steps):
            aug_type = '_masked' if masked else ''
            for frame in range(self.num_frames):
                if frame == 0 or self.save_all_frames:
                    torchvision.utils.save_image(
                        obs[0][3 * frame:3 * frame + 3] / 255.,
                        os.path.join(self.dir_name, f'step{step}_frame{frame}_augmented{aug_type}_obs.png'))



def log_soft_mask_stats(agent, soft_mask, logger: Logger, training_step: int, _test_env: str):
    avg = soft_mask.mean()
    logger.log(f'eval_soft_mask/avg{_test_env}', avg, training_step)
    logger.log(f'eval_soft_mask/std{_test_env}', soft_mask.std(), training_step)
    logger.log(f'eval_soft_mask/min{_test_env}', soft_mask.min(), training_step)
    logger.log(f'eval_soft_mask/max{_test_env}', soft_mask.max(), training_step)
    logger.log(f'eval_soft_mask/median{_test_env}', soft_mask.median(), training_step)
    # check what the quantile of the avg is
    soft_mask_sorted = soft_mask.view(-1).sort()[0]
    avg_idx = (torch.abs(soft_mask_sorted - avg)).argmin()
    quantile = avg_idx / len(soft_mask_sorted)
    logger.log(f'eval_soft_mask/quantile_of_avg{_test_env}', quantile, training_step)
    # Log statistics of the masker threshold if `quantile`
    if agent.masker.thresholds:
        logger.log(f'eval_soft_mask/threshold{_test_env}', torch.stack(agent.masker.thresholds).mean(), training_step)
        # Clear the list of thresholds
        agent.masker.thresholds = []
