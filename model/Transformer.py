"""
This implementation is adapted from PoseFormer: https://github.com/zczcwh/PoseFormer
"""
import torch
import torch.nn as nn
from einops import rearrange
from functools import partial

from model.Attention import AttentionBlock

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, num_frame=124, num_joints=8, in_chans=2, joint_embed_dim=12, num_heads=2, 
                mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.):
        """
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            joint_embed_dim (int): the dimension of joint embedding
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
        """
        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        temporal_embed_dim = joint_embed_dim * num_joints
        out_dim = 1

        ### joint embedding
        self.joint_embedding = nn.Linear(in_chans, joint_embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        ### positional encoding for spatial and temporal attention (learnable)
        self.spatial_positional_enc = nn.Parameter(torch.zeros(1, num_joints, joint_embed_dim))
        self.temporal_positional_enc = nn.Parameter(torch.zeros(1, num_frame, temporal_embed_dim))
        
        ### spatial attention block
        self.spatial_block = AttentionBlock(dim=joint_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)

        ### temporal attention block
        self.temporal_block = AttentionBlock(dim=temporal_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)

        self.spatial_norm = norm_layer(joint_embed_dim)
        self.temporal_norm = norm_layer(temporal_embed_dim)
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(temporal_embed_dim),
            nn.Linear(temporal_embed_dim, 8),
            nn.ReLU(),
            nn.Linear(8, out_dim)
        )

    def spatial_forward(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.joint_embedding(x)
        x += self.spatial_positional_enc
        x = self.pos_drop(x)

        x = self.spatial_block(x)
        x = self.spatial_norm(x)

        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def temporal_forward(self, x):
        b = x.shape[0]

        x += self.temporal_positional_enc
        x = self.pos_drop(x)
        
        x = self.temporal_block(x)
        x = self.temporal_norm(x)
        
        x = self.weighted_mean(x)
        x = x.view(b, 1, -1)
        return x

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        
        x = self.spatial_forward(x)
        x = self.temporal_forward(x)
        
        x = self.regression_head(x)
        
        x = x.view(-1)
        return x