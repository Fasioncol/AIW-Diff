import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import VisionTransformer, IB_MLP, Normalize, nonlinearity
from pytorch_wavelets import DWTForward, DWTInverse

from .wavelet_ms_down import WaveletMSDown

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class WaveletGatingModule(nn.Module):
    def __init__(self, in_channels, temb_channels, gate_bias=1.0):
        super().__init__()
        self.gate_bias = gate_bias
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_channels * 2 + temb_channels, in_channels)
        )

    def forward(self, forig, fwave, temb):
        if temb.dim() == 2:
             temb_proj = temb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, forig.shape[-2], forig.shape[-1])
        else:
             temb_proj = temb

        combined_features = torch.cat([forig, fwave, temb_proj], dim=1)
        gate_logits = self.mlp(combined_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        g = torch.sigmoid(gate_logits - self.gate_bias)
        return (1 - g) * forig + g * fwave

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, config):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)

        self.complexity_proj = torch.nn.Conv2d(1, out_channels, kernel_size=1, stride=1, padding=0)
        self.complexity_strength = getattr(config.model, 'complexity_strength', 1.0)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        if config.model.wavelet.use_wavelet:
            self.dwt = DWTForward(J=1, wave=config.model.wavelet.wavelet_type, mode='zero')
            self.idwt = DWTInverse(wave=config.model.wavelet.wavelet_type, mode='zero')

            self.subband_attention = nn.Sequential(
                nn.SiLU(),
                nn.Conv2d(out_channels * 4, out_channels * 4, 1),
                nn.SiLU()
            )
            self.temb_to_wavelet = nn.Linear(temb_channels, out_channels * 4)

            self.wavelet_gating = WaveletGatingModule(
                out_channels, temb_channels,
                gate_bias=config.model.wavelet.gate_bias
            )

    def forward(self, x, temb, complexity_map):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        forig = self.conv1(h)

        if self.config.model.wavelet.use_wavelet and forig.shape[-1] > 1 and forig.shape[-1] % 2 == 0:
            ll, yh_list = self.dwt(forig)
            yh = yh_list[0]

            b, c, _, h_w, w_w = yh.shape
            yh_flat = yh.view(b, c * 3, h_w, w_w)
            all_bands = torch.cat([ll, yh_flat], dim=1)

            temb_wavelet = self.temb_to_wavelet(nonlinearity(temb)).unsqueeze(-1).unsqueeze(-1)
            attended_bands = self.subband_attention(all_bands + temb_wavelet)

            attended_ll, attended_yh_flat = torch.split(attended_bands, [c, c * 3], dim=1)
            attended_yh = attended_yh_flat.view(b, c, 3, h_w, w_w)

            fwave = self.idwt((attended_ll, [attended_yh]))

            h_fused = self.wavelet_gating(forig, fwave, temb)
            h = h_fused
        else:
            h = forig

        temb_feat = self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = h + temb_feat

        if complexity_map is not None:
            if complexity_map.shape[-2:] != h.shape[-2:]:
                comp_map_resized = F.interpolate(complexity_map,
                                                 size=h.shape[-2:],
                                                 mode='bilinear',
                                                 align_corners=False)
            else:
                comp_map_resized = complexity_map

            comp_feat = self.complexity_proj(comp_map_resized)
            h = h + (self.complexity_strength * comp_feat)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_

def _ckpt_call(fn, *args):
    if torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)

def _call_resblock(block, h, temb, ent):
    return block(h, temb, ent)

def _call_attn(block, h):
    return block(h)

def _call_down(down, h):
    return down(h)

def _call_vit(vit, h):
    return vit(h)

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch = config.model.ch, config.model.out_ch
        ch_mult = tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch,
                                         dropout=dropout, config=config))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:

                down.downsample = WaveletMSDown(
                    in_channels=block_in,
                    out_channels=block_in,
                    wavelet_type=config.model.wavelet.wavelet_type
                )

                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch,
                                       dropout=dropout, config=config)

        vit_patch_size = config.model.vit.patch_size
        self.mid.vit_1 = VisionTransformer(
            input_size=(curr_res, curr_res),
            patch_size=vit_patch_size,
            embed_dim=block_in, num_heads=config.model.vit.num_heads,
            depth=config.model.vit.depth, channels=block_in
        )
        vit_1_output_res = (curr_res // vit_patch_size)
        ib_input_features = block_in * (vit_1_output_res ** 2)
        self.mid.ib_mlp = IB_MLP(
            in_features=ib_input_features, z_dim=config.model.ib_mlp.z_dim
        )
        self.mid.vit_2 = VisionTransformer(
            input_size=(vit_1_output_res, vit_1_output_res),
            patch_size=vit_patch_size,
            embed_dim=block_in, num_heads=config.model.vit.num_heads,
            depth=config.model.vit.depth, channels=block_in
        )
        if vit_patch_size > 1:
            self.mid.upsample = nn.Upsample(scale_factor=curr_res // (vit_1_output_res // vit_patch_size), mode='bilinear', align_corners=False)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch,
                                       dropout=dropout, config=config)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(in_channels=block_in + skip_in, out_channels=block_out, temb_channels=self.temb_ch,
                                dropout=dropout, config=config))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        self.conv_out_clean = torch.nn.Conv2d(block_in, self.in_channels // 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, complexity_map):
        assert x.shape[2] == x.shape[3] == self.resolution

        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        ent_maps = [complexity_map]
        for _ in range(self.num_resolutions - 1):
            ent_maps.append(
                torch.nn.functional.avg_pool2d(ent_maps[-1], kernel_size=2, stride=2)
            )

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = _ckpt_call(_call_resblock, self.down[i_level].block[i_block], hs[-1], temb, None)

                if len(self.down[i_level].attn) > 0:
                    h = _ckpt_call(_call_attn, self.down[i_level].attn[i_block], h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(_ckpt_call(_call_down, self.down[i_level].downsample, hs[-1]))

        h = hs[-1]

        h = _ckpt_call(_call_resblock, self.mid.block_1, h, temb, None)
        h_vit_1 = _ckpt_call(_call_vit, self.mid.vit_1, h)
        h_ib, mu, log_var = self.mid.ib_mlp(h_vit_1)
        h_vit_2 = _ckpt_call(_call_vit, self.mid.vit_2, h_ib)

        h = h_vit_2
        if self.config.model.vit.patch_size > 1:
            h = self.mid.upsample(h)

        h = _ckpt_call(_call_resblock, self.mid.block_2, h, temb, None)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = _ckpt_call(
                    _call_resblock,
                    self.up[i_level].block[i_block],
                    torch.cat([h, hs.pop()], dim=1),
                    temb,
                    ent_maps[i_level],
                )
                if len(self.up[i_level].attn) > 0:
                    h = _ckpt_call(_call_attn, self.up[i_level].attn[i_block], h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)

        predicted_noise = self.conv_out(h)
        predicted_clean_image = self.conv_out_clean(h)

        return predicted_noise, mu, log_var, predicted_clean_image
