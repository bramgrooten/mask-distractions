import numpy as np
import torch
import torch.nn.functional as F
import utils
from algorithms.vit.svea_vit import SVEA_ViT
import algorithms.modules as m
from video import AugmentationRecorder
from augmentations import strong_augment


class MaDi_ViT(SVEA_ViT):
	"""MaDi: Masking Distractions for Generalization in Reinforcement Learning
	Vision Transformer backbone in this version"""
	def __init__(self, obs_shape, action_shape, args, aug_recorder: AugmentationRecorder):
		super().__init__(obs_shape, action_shape, args)
		self.masker = m.MaskerNet(obs_shape, args).cuda()
		self.masker_optimizer = torch.optim.Adam(
			self.masker.parameters(), lr=args.masker_lr, betas=(args.masker_beta, 0.999)
		)
		self.num_masks = args.frame_stack
		self.aug_recorder = aug_recorder

	def apply_mask(self, obs, test_env=False):
		# obs: tensor shaped as (B, 9, H, W)
		frames = obs.chunk(self.num_masks, dim=1)  # frames: list of tensors [ (B,3,H,W) , (B,3,H,W) , (B,3,H,W) ]
		frames_cat = torch.cat(frames, dim=0)  # concat in batch dim. frames_cat: tensor shaped (B*3, 3, H, W)
		masks_cat = self.masker(frames_cat, test_env=test_env)  # apply MaskerNet just once. masks_cat: (B*3, 1, H, W)
		masks = masks_cat.chunk(self.num_masks, dim=0)  # split the batch dim back into channel dim. masks: list of tensors [ (B,1,H,W) , (B,1,H,W) , (B,1,H,W) ]
		masked_frames = [m * f for m, f in zip(masks, frames)]  # element-wise multiplication. uses broadcasting. masked_frames: list of tensors [ (B,3,H,W) , (B,3,H,W) , (B,3,H,W) ]
		return torch.cat(masked_frames, dim=1)  # concat in channel dim. returns: tensor shaped (B, 9, H, W)

	def select_action(self, obs, test_env=False):
		_obs = self._obs_to_input(obs)
		_obs = self.apply_mask(_obs, test_env)
		with torch.no_grad():
			mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
		return mu.cpu().data.numpy().flatten()

	def sample_action(self, obs):
		_obs = self._obs_to_input(obs)
		_obs = self.apply_mask(_obs)
		with torch.no_grad():
			mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
		return pi.cpu().data.numpy().flatten()

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			next_obs = self.apply_mask(next_obs)
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		obs_aug = strong_augment(obs, self.augment, self.overlay_alpha)
		self.aug_recorder.record(obs_aug, step)

		if self.svea_alpha == self.svea_beta:
			obs = utils.cat(obs, obs_aug)
			obs = self.apply_mask(obs)
			self.aug_recorder.record(obs[obs.shape[0] // 2:], step, masked=True)
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		else:
			# chance to speedup: apply_mask just once here as well
			# just concat(obs, obs_aug), apply, then split back again
			# actually: could probably do the same with the 2 critic forward passes in regular SVEA
			obs = self.apply_mask(obs)
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.svea_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
			obs_aug = self.apply_mask(obs_aug)
			self.aug_recorder.record(obs_aug, step, masked=True)
			current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
			critic_loss += self.svea_beta * \
				(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)

		self.masker_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		self.masker_optimizer.step()

	def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
		obs = self.apply_mask(obs)
		_, pi, log_pi, log_std = self.actor(obs, detach=True)
		actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

		if L is not None:
			L.log('train_actor/loss', actor_loss, step)
			entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		if update_alpha:
			self.log_alpha_optimizer.zero_grad()
			alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

			if L is not None:
				L.log('train_alpha/loss', alpha_loss, step)
				L.log('train_alpha/value', self.alpha, step)

			alpha_loss.backward()
			self.log_alpha_optimizer.step()
