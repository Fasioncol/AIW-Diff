import torch
import torch.nn as nn
import numpy as np

def nonlinearity(x):
    return nn.SiLU()(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, 1, embed_dim]), pos_embed], axis=1)

    return torch.from_numpy(pos_embed).float().unsqueeze(0)

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 4 == 0, "Embed dim must be divisible by 4 for 2D sin/cos"

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= (embed_dim // 2. - 1.)
    omega = 1. / (10000**omega)

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(drop)
        )

        self.rezero_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.rezero_alpha * attn_output

        mlp_output = self.mlp(self.norm2(x))
        x = x + self.rezero_alpha * mlp_output
        return x

class VisionTransformer(nn.Module):
    def __init__(self, input_size, patch_size, embed_dim, num_heads, depth=12, channels=1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        num_patches_h = input_size[0] // patch_size
        num_patches_w = input_size[1] // patch_size
        self.num_patches = num_patches_h * num_patches_w

        self.proj = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        pos_embed_tensor = get_2d_sincos_pos_embed(
            embed_dim=embed_dim,
            grid_size_h=num_patches_h,
            grid_size_w=num_patches_w
        )
        self.register_buffer('pos_embed', pos_embed_tensor)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x_proj = self.proj(x)

        _, E, H_new, W_new = x_proj.shape

        x = x_proj.flatten(2).transpose(1, 2)

        if x.shape[1] != self.pos_embed.shape[1]:
             raise ValueError(f"ViT: number of patches mismatch. Input N={x.shape[1]}, pos_embed N={self.pos_embed.shape[1]}")

        x = x + self.pos_embed.to(x.device)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x = x.transpose(1, 2).reshape(B, self.embed_dim, H_new, W_new)
        return x

class IB_MLP(nn.Module):
    def __init__(self, in_features, z_dim):
        super().__init__()
        self.encoder_mu = nn.Linear(in_features, z_dim)
        self.encoder_log_var = nn.Linear(in_features, z_dim)
        self.decoder = nn.Linear(z_dim, in_features)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        B, C, H, W = x.shape

        x_flat = x.reshape(B, -1)

        mu = self.encoder_mu(x_flat)
        log_var = self.encoder_log_var(x_flat)
        z = self.reparameterize(mu, log_var)
        x_recon_flat = self.decoder(z)

        x_recon = x_recon_flat.view(B, C, H, W)

        return x_recon, mu, log_var
