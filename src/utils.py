import argparse
from typing import Union
import torch
import numpy as np
import os
import glob
import json
import random
import augmentations
import subprocess
from datetime import datetime
from env.wrappers import make_env


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        'timestamp': str(datetime.now()),
        'args': vars(args),
        'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
    }
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


def load_config(key=None):
    dmcgb_datasets = os.environ.get('DMCGB_DATASETS')
    assert dmcgb_datasets is not None, 'DMCGB_DATASETS not set. Use `export DMCGB_DATASETS="/path/to/datasets"`'
    return [dmcgb_datasets]


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def listdir(dir_path, filetype='jpg', sort=True):
    fpath = os.path.join(dir_path, f'*.{filetype}')
    fpaths = glob.glob(fpath, recursive=True)
    if sort:
        return sorted(fpaths)
    return fpaths


def prefill_memory(obses, capacity, obs_shape):
    """Reserves memory for replay buffer"""
    c, h, w = obs_shape
    for _ in range(capacity):
        frame = np.ones((3, h, w), dtype=np.uint8)
        obses.append(frame)
    return obses


class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, prefill=True):
        self.capacity = capacity
        self.batch_size = batch_size

        self._obses = []
        if prefill:
            self._obses = prefill_memory(self._obses, capacity, obs_shape)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.prefill = prefill
        self.obs_shape = obs_shape
        self.action_shape = action_shape

    def reset(self):
        self._obses = []
        if self.prefill:
            self._obses = prefill_memory(self._obses, self.capacity, self.obs_shape)
        self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        obses = (obs, next_obs)
        if self.idx >= len(self._obses):
            self._obses.append(obses)
        else:
            self._obses[self.idx] = (obses)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _get_idxs(self, n=None):
        if n is None:
            n = self.batch_size
        return np.random.randint(
            0, self.capacity if self.full else self.idx, size=n
        )

    def _encode_obses(self, idxs):
        obses, next_obses = [], []
        for i in idxs:
            obs, next_obs = self._obses[i]
            obses.append(np.array(obs, copy=False))
            next_obses.append(np.array(next_obs, copy=False))
        return np.array(obses), np.array(next_obses)

    def sample_soda(self, n=None):
        idxs = self._get_idxs(n)
        obs, _ = self._encode_obses(idxs)
        return torch.as_tensor(obs).cuda().float()

    def __sample__(self, n=None):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        return obs, actions, rewards, next_obs, not_dones

    def sample_curl(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        pos = augmentations.random_crop(obs.clone())
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones, pos

    def sample_drq(self, n=None, pad=4):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.random_shift(obs, pad)
        next_obs = augmentations.random_shift(next_obs, pad)

        return obs, actions, rewards, next_obs, not_dones

    def sample_svea(self, n=None, pad=4):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.random_shift(obs, pad)

        return obs, actions, rewards, next_obs, not_dones

    def sample_vit(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.vit_crop(obs)
        next_obs = augmentations.vit_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones

    def sample(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones

    def sample_no_aug(self, n=None):
        return self.__sample__(n=n)


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0]//3

    def frame(self, i):
        return self._force()[i*3:(i+1)*3]


def count_parameters(net, as_int=False):
    """Returns total number of params in a network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f'{count:,}'


def make_envs(args):
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode=args.train_env_mode,
        intensity=args.distracting_cs_intensity
    )
    train_env_eval = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode=args.train_env_mode,
        intensity=args.distracting_cs_intensity
    )
    test_envs = []
    test_modes = []
    if args.eval_mode in ['color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs']:
        test_env = make_env(
            domain_name=args.domain_name,
            task_name=args.task_name,
            seed=args.seed + 42,
            episode_length=args.episode_length,
            action_repeat=args.action_repeat,
            image_size=args.image_size,
            mode=args.eval_mode,
            intensity=args.distracting_cs_intensity
        )
        test_envs.append(test_env)
        test_modes.append(args.eval_mode)
    elif args.eval_mode == 'both':
        for eval_mode in ["video_easy", "video_hard"]:
            test_env = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed + 42,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode=eval_mode,
                intensity=0.0,
            )
            test_envs.append(test_env)
            test_modes.append(eval_mode)
    elif args.eval_mode == 'hard_dcs':
        for eval_mode in ["video_hard", "distracting_cs"]:
            test_env = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed + 42,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode=eval_mode,
                intensity=args.distracting_cs_intensity,
            )
            test_envs.append(test_env)
            test_modes.append(eval_mode)
    elif args.eval_mode == 'all':
        for eval_mode in ["video_easy", "video_hard", "distracting_cs"]:
            test_env = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed + 42,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode=eval_mode,
                intensity=args.distracting_cs_intensity,
            )
            test_envs.append(test_env)
            test_modes.append(eval_mode)
    elif args.eval_mode == 'four':
        for eval_mode in ["color_hard", "video_easy", "video_hard", "distracting_cs"]:
            test_env = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed + 42,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode=eval_mode,
                intensity=args.distracting_cs_intensity,
            )
            test_envs.append(test_env)
            test_modes.append(eval_mode)
    elif args.eval_mode == 'intens':
        for eval_mode in ["video_easy", "video_hard"]:
            test_env = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed + 42,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode=eval_mode,
                intensity=0.0,
            )
            test_envs.append(test_env)
            test_modes.append(eval_mode)
        for intensity in [0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
            test_env = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed + 42,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode="distracting_cs",
                intensity=intensity,
            )
            test_envs.append(test_env)
            intensity_str = str(intensity).split(".")[1]
            test_modes.append(f"distracting_cs{intensity_str}")
    elif args.eval_mode is None:
        test_envs = None
    else:
        raise ValueError(f"Unknown eval mode {args.eval_mode}")
    return env, train_env_eval, test_envs, test_modes


def logging_sgqn(agent, i, episode_step, step, obs, torch_obs, torch_action, action, test_mode):
    """Some extra logging from the SGQN code."""
    # log in tensorboard 15th step
    if i == 0 and episode_step in [15, 16, 17, 18] and step > 0:
        _obs = agent._obs_to_input(obs)
        torch_obs.append(_obs)
        torch_action.append(torch.tensor(action).to(_obs.device).unsqueeze(0))
        if episode_step == 18:
            prefix = "eval" if test_mode is None else test_mode
            agent.log_tensorboard(
                torch.cat(torch_obs, 0),
                torch.cat(torch_action, 0),
                step, prefix=prefix,
            )
    # attrib_grid = make_obs_grad_grid(torch.sigmoid(mask))
    # agent.writer.add_image(
    #     prefix + "/smooth_attrib", attrib_grid, global_step=step
    # )


def save_img(obs, path='obs.png'):
    import torchvision
    torchvision.utils.save_image(obs[0][:3] / 255., path)


def set_experiment_name(args):
    encoder = f'-vit' if args.encoder != 'cnn' else ''
    aug_type = f'_aug-{args.augment}' if args.augment != 'none' else ''
    batch = f'_batch{args.batch_size}' if args.batch_size != 64 else ''
    train = f'_train-{args.train_env_mode}' if args.train_env_mode != 'clean' else ''
    eval_mode = f'_eval-{args.eval_mode}' if args.eval_mode is not None else ''
    dcs_intensity = f'(intens{args.distracting_cs_intensity})' if args.distracting_cs_intensity != 0.1 and args.eval_mode in ['distracting_cs', 'all'] else ''
    sgqn_quantile = f'_q{args.sgqn_quantile}' if args.algorithm == 'sgqn' else ''
    c_weight_decay = f'_cWeightDecay{args.critic_weight_decay}' if args.critic_weight_decay != 0 else ''
    masker_layers = f'_mLay{args.masker_num_layers}' if args.algorithm == 'madi' and args.masker_num_layers != 3 else ''
    overlay_alpha = f'_alpha={args.overlay_alpha}' if args.overlay_alpha != 0.5 and args.augment == 'overlay' else ''
    svea_alpha = f'_sveaA{args.svea_alpha}' if args.svea_alpha != 0.5 else ''
    svea_beta = f'_sveaB{args.svea_beta}' if args.svea_beta != 0.5 else ''
    env = f"{args.domain_name}-{args.task_name}"

    name = f"{env}_{args.algorithm}{encoder}{svea_alpha}{svea_beta}{aug_type}" \
           f"{overlay_alpha}{c_weight_decay}{batch}{train}{eval_mode}{dcs_intensity}" \
           f"{sgqn_quantile}{masker_layers}" \
           f"_seed{args.seed}"
    return name


def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
