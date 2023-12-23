import numpy as np
import torch
from copy import deepcopy
import utils
import algorithms.modules as m
from algorithms.sac import SAC


class SAC_ViT(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
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

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_no_aug()

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()
