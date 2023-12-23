import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--domain_name', default='walker')
    parser.add_argument('--task_name', default='walk')
    parser.add_argument('--train_env_mode', default='clean', type=str)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--action_repeat', default=4, type=int, help='this is the frame skip')
    parser.add_argument('--episode_length', default=1000, type=int)

    # eval
    parser.add_argument('--eval_mode', default='none', type=str, help='modes: color_easy, color_hard, video_easy, video_hard, distracting_cs. Also: none, both, all, four, intens. See utils.make_env()')
    parser.add_argument('--save_freq', default='500k', type=str, help='how often to save the agent models')
    parser.add_argument('--eval_freq', default='10k', type=str)
    parser.add_argument('--eval_episodes', default=20, type=int)
    parser.add_argument('--distracting_cs_intensity', default=0.1, type=float)

    # agent
    parser.add_argument('--algorithm', default='madi', type=str)
    parser.add_argument('--encoder', default='cnn', type=str, choices=['cnn', 'vit'], help='either `cnn` or `vit`. ViT only implemented for MaDi and SVEA.')
    parser.add_argument('--train_steps', default='500k', type=str)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--batch_size', default=64, type=int, help='batch size in paper: CNN-128, ViT-256')
    parser.add_argument('--hidden_dim', default=1024, type=int)

    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)

    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    parser.add_argument("--critic_weight_decay", default=0, type=float, help="weight decay for critic optimizer, used in SGQN")

    # cnn architecture
    parser.add_argument('--num_shared_layers', default=11, type=int)
    parser.add_argument('--num_head_layers', default=0, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--projection_dim', default=100, type=int)
    parser.add_argument('--encoder_tau', default=0.05, type=float)

    # vit architecture
    parser.add_argument('--patch_size', default=8, type=int)
    parser.add_argument('--embed_dim', default=128, type=int)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--mlp_ratio', default=1., type=float)
    parser.add_argument('--qvk_bias', default=False, action='store_true')

    # entropy maximization
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)

    # augmentation
    parser.add_argument('--augment', default='none', type=str, choices=['none', 'conv', 'overlay', 'splice'], help='Either `none`, `conv`, `overlay`, or `splice`. Type of strong augmentation used in the algorithms MaDi, SGQN, SVEA, SODA.')
    parser.add_argument('--overlay_alpha', default=0.5, type=float, help='Opacity of the overlay augmentation')
    parser.add_argument('--save_aug', default=False, action='store_true', help='save the augmented observations during the critic update in MaDi')
    parser.add_argument('--save_aug_every', default=1e4, type=int, help='save augmented images every n steps')
    parser.add_argument('--save_all_frames', default=False, action='store_true', help='whether to save each frame of the observation or just the first one')

    # auxiliary tasks
    parser.add_argument('--aux_lr', default=1e-3, type=float, help='learning rate for auxiliary tasks. Default is 1e-3, except for SGQN & SODA: 3e-4 is used there.')  # see below
    parser.add_argument('--aux_beta', default=0.9, type=float)
    parser.add_argument('--aux_update_freq', default=2, type=int)

    # soda
    parser.add_argument('--soda_batch_size', default=256, type=int)
    parser.add_argument('--soda_tau', default=0.005, type=float)

    # svea
    parser.add_argument('--svea_alpha', default=0.5, type=float)
    parser.add_argument('--svea_beta', default=0.5, type=float)
    parser.add_argument('--svea_weight_decay', default=False, action='store_true', help='weight decay for encoder, used in SVEA-ViT only')

    # sgqn
    parser.add_argument("--sgqn_quantile", default=0.95, type=float)  # Table 3 of SGQN paper. See below.
    parser.add_argument("--svea_contrastive_coeff", default=0.1, type=float)
    parser.add_argument("--svea_norm_coeff", default=0.1, type=float)
    parser.add_argument("--attrib_coeff", default=0.25, type=float)
    parser.add_argument("--consistency", default=1, type=int)

    # madi
    parser.add_argument('--save_mask', default=False, action='store_true', help='save the masks used for MaDi')
    parser.add_argument('--masker_lr', default=1e-3, type=float)
    parser.add_argument('--masker_beta', default=0.9, type=float)
    parser.add_argument('--masker_num_layers', default=3, type=int)
    parser.add_argument('--masker_num_filters', default=32, type=int)
    parser.add_argument('--mask_type', default='soft', type=str, choices=['soft', 'hard', 'mixed'], help='either: `soft` (continuous), `hard` (binary), or `mixed`. If `hard`, then the mask is a binary mask. If `soft`, then the mask has continuous values between 0 and 1. If `mixed`, then the training env is soft masked and the test env is hard masked.')
    parser.add_argument('--mask_threshold', default=0.5, type=float, help='initial threshold (or quantile) for the mask. Only used if `mask_type` is not `soft`.')
    parser.add_argument('--mask_threshold_type', default='fix', type=str, choices=['fix', 'avg', 'quantile'], help='Only used if `mask_type` is not `soft`.')

    # wandb
    parser.add_argument('--wandb_mode', default='online', type=str, help='`online` or `offline` or `disabled`', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--tag', default='default_tag', type=str, help='tag to identify the run on wandb')
    parser.add_argument('--console_log', default='none_set', type=str, help='file to log the console output to, to have as arg on wandb')

    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_vid_every', default=1e5, type=int, help='record a video of an episode every n steps')

    args = parser.parse_args()

    algorithms = {'sac', 'sac_aug', 'drq', 'rad', 'curl', 'pad', 'soda', 'svea', 'sgqn', 'madi'}
    env_modes = {'clean', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs',
                 'none', 'both', 'hard_dcs', 'all', 'four', 'intens'}
    intensities = {0., 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
    assert args.algorithm in algorithms, f'specified algorithm "{args.algorithm}" is not supported'
    assert args.train_env_mode in env_modes, f'specified mode "{args.train_env_mode}" is not supported'
    assert args.eval_mode in env_modes, f'specified mode "{args.eval_mode}" is not supported'
    assert args.distracting_cs_intensity in intensities, f'distracting_cs has only been implemented for intensities: {intensities}'
    assert args.log_dir is not None, 'must provide a log directory for experiment'

    args.train_steps = int(args.train_steps.replace('k', '000'))
    args.save_freq = int(args.save_freq.replace('k', '000'))
    args.eval_freq = int(args.eval_freq.replace('k', '000'))
    args.eval_mode = None if args.eval_mode == 'none' else args.eval_mode

    if args.encoder == 'cnn' and args.algorithm in {'rad', 'curl', 'pad', 'soda'}:
        args.image_size = 100
        args.image_crop_size = 84
    elif args.encoder == 'cnn' and args.algorithm not in {'rad', 'curl', 'pad', 'soda'}:
        args.image_size = 84
        args.image_crop_size = 84
    elif args.encoder == 'vit' and args.algorithm in {'rad'}:
        args.image_size = 100
        args.image_crop_size = 96
    elif args.encoder == 'vit' and args.algorithm in {'madi', 'svea', 'sac', 'drq'}:
        args.image_size = 96
        args.image_crop_size = 96
    else:
        raise NotImplementedError(f'Algorithm: {args.algorithm} not yet implemented for encoder: {args.encoder}')

    ############ Watch out for these default settings below!
    if args.algorithm == 'sgqn':
        # see Table 3 of SGQN paper: https://arxiv.org/pdf/2209.09203.pdf
        args.aux_lr = 3e-4
        args.critic_weight_decay = 1e-5
        if args.domain_name in ['walker', 'finger', 'cheetah', 'hopper', 'humanoid']:
            args.sgqn_quantile = 0.95  # (big agents, less masking)
        else:
            args.sgqn_quantile = 0.98  # (small agents, more masking)
        print(f'Using sgqn_quantile: {args.sgqn_quantile} for domain: {args.domain_name}')
    elif args.algorithm == 'soda':
        args.aux_lr = 3e-4

    return args
