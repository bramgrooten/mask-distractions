from algorithms.sac_no_aug import SACNoAug
from algorithms.sac import SAC
from algorithms.drq import DrQ
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.svea import SVEA
from algorithms.sgqn import SGQN
from algorithms.madi import MaDi
from algorithms.vit.sac_vit import SAC_ViT
from algorithms.vit.drq_vit import DrQ_ViT
from algorithms.vit.rad_vit import RAD_ViT
from algorithms.vit.svea_vit import SVEA_ViT
from algorithms.vit.madi_vit import MaDi_ViT


algorithm = {
	'sac': SACNoAug,
	'sac_aug': SAC,  # uses light augmentation (random crop)
	'drq': DrQ,
	'rad': RAD,
	'curl': CURL,
	'pad': PAD,
	'soda': SODA,
	'svea': SVEA,
	'sgqn': SGQN,
	'madi': MaDi,
}

vit_algorithm = {
	'sac': SAC_ViT,
	'drq': DrQ_ViT,
	'rad': RAD_ViT,
	'svea': SVEA_ViT,
	'madi': MaDi_ViT,
}


def make_agent(obs_shape, action_shape, args, agent_args: dict):
	if args.encoder == 'cnn':
		return algorithm[args.algorithm](obs_shape, action_shape, args, **agent_args)
	elif args.encoder == 'vit':
		return vit_algorithm[args.algorithm](obs_shape, action_shape, args, **agent_args)
	else:
		raise ValueError(f'unknown encoder "{args.encoder}"')
