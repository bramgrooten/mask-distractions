import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
from algorithms.sac import SAC
from augmentations import strong_augment


class SVEA_ViT(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta
		self.augment = args.augment
		self.overlay_alpha = args.overlay_alpha

		self.discount = args.discount
		self.critic_tau = args.critic_tau
		self.encoder_tau = args.encoder_tau
		self.actor_update_freq = args.actor_update_freq
		self.critic_target_update_freq = args.critic_target_update_freq

		shared = m.SharedTransformer(
			obs_shape,
			args.patch_size,
			args.embed_dim,
			args.depth,
			args.num_heads,
			args.mlp_ratio,
			args.qvk_bias
		).cuda()
		head = m.HeadCNN(shared.out_shape, args.num_head_layers, args.num_filters).cuda()
		actor_encoder = m.Encoder(
			shared,
			head,
			m.RLProjection(head.out_shape, args.projection_dim)
		)
		critic_encoder = m.Encoder(
			shared,
			head,
			m.RLProjection(head.out_shape, args.projection_dim)
		)

		self.actor = m.Actor(actor_encoder, action_shape, args.hidden_dim, args.actor_log_std_min, args.actor_log_std_max).cuda()
		self.critic = m.Critic(critic_encoder, action_shape, args.hidden_dim).cuda()
		self.critic_target = deepcopy(self.critic)

		self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
		self.log_alpha.requires_grad = True
		self.target_entropy = -np.prod(action_shape)

		self.actor_optimizer = torch.optim.Adam(
			self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
		)
		if args.svea_weight_decay:
			self.critic_optimizer = torch.optim.AdamW(
				[
					{'params': self.critic.encoder.shared.parameters(), 'weight_decay': 1e-2},
					{'params': self.critic.encoder.head.parameters()},
					{'params': self.critic.encoder.projection.parameters()},
					{'params': self.critic.Q1.parameters()},
					{'params': self.critic.Q2.parameters()}
				], lr=args.critic_lr, betas=(args.critic_beta, 0.999), weight_decay=0
			)
		else:
			self.critic_optimizer = torch.optim.Adam(
				self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
			)
		self.log_alpha_optimizer = torch.optim.Adam(
			[self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
		)
		self.train()

		print('Shared ViT:', utils.count_parameters(shared))
		print('Head:', utils.count_parameters(head))
		print('Projection:', utils.count_parameters(critic_encoder.projection))
		print('Critic: 2x', utils.count_parameters(self.critic.Q1))

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		obs_aug = strong_augment(obs, self.augment, self.overlay_alpha)

		if self.svea_alpha == self.svea_beta:
			obs = utils.cat(obs, obs_aug)
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.svea_alpha * (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
			current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
			critic_loss += self.svea_beta * (F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))
		
		if L is not None:
			L.log('train_critic/loss', critic_loss, step)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_vit()

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()
