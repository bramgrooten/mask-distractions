from functools import partial
import torch
import torch.nn as nn
from algorithms.vit.layers import to_2tuple, trunc_normal_, lecun_normal_


class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.fc2(x)
		return x


class Attention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = qk_scale or head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.proj = nn.Linear(dim, dim)

	def forward(self, x):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		return x


class Block(nn.Module):
	def __init__(self, dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None,
				 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.attn = Attention(
			dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

	def forward(self, x):
		x = x + self.attn(self.norm1(x))
		x = x + self.mlp(self.norm2(x))
		return x


class PatchEmbed(nn.Module):
	def __init__(self, img_size=84, patch_size=14, in_chans=3, embed_dim=64, norm_layer=None):
		super().__init__()
		img_size = to_2tuple(img_size)
		patch_size = to_2tuple(patch_size)
		self.img_size = img_size
		self.patch_size = patch_size
		self.patch_grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
		self.num_patches = self.patch_grid[0] * self.patch_grid[1]
		print('Num patches:', self.num_patches)

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
		self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

	def forward(self, x):
		B, C, H, W = x.shape
		x = self.proj(x).flatten(2).transpose(1, 2)
		x = self.norm(x)
		return x


class VisionTransformer(nn.Module):
	def __init__(self, img_size=84, patch_size=14, in_chans=3, embed_dim=64, depth=4, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None):
		super().__init__()
		self.embed_dim = embed_dim
		norm_layer = partial(nn.LayerNorm, eps=1e-6)

		self.patch_embed = PatchEmbed(
			img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
		num_patches = self.patch_embed.num_patches

		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

		self.blocks = nn.Sequential(*[
			Block(
				dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
				norm_layer=norm_layer, act_layer=nn.GELU)
			for i in range(depth)])
		self.norm = norm_layer(embed_dim)

		# Weight init
		trunc_normal_(self.pos_embed, std=.02)
		trunc_normal_(self.cls_token, std=.02)
		self.apply(_init_vit_weights)

	@torch.jit.ignore
	def no_weight_decay(self):
		return {'pos_embed', 'cls_token'}

	def forward(self, x):
		x = self.patch_embed(x)
		cls_token = self.cls_token.expand(x.size(0), -1, -1)
		x = torch.cat((cls_token, x), dim=1)
		x = x + self.pos_embed
		x = self.blocks(x)
		x = self.norm(x)
		return x[:, 0]


def _init_vit_weights(m, n: str = '', head_bias: float = 0.):
	if isinstance(m, nn.Linear):
		if n.startswith('head'):
			nn.init.zeros_(m.weight)
			nn.init.constant_(m.bias, head_bias)
		elif n.startswith('pre_logits'):
			lecun_normal_(m.weight)
			nn.init.zeros_(m.bias)
		else:
			trunc_normal_(m.weight, std=.02)
			if m.bias is not None:
				nn.init.zeros_(m.bias)
	elif isinstance(m, nn.LayerNorm):
		nn.init.zeros_(m.bias)
		nn.init.ones_(m.weight)
