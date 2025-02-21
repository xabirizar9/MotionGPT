# Partially from https://github.com/Mael-zys/T2M-GPT

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from .tools.resnet import Resnet1D
from .tools.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from collections import OrderedDict


class VQVae(nn.Module):

    def __init__(self,
                 nfeats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 codebook_logger: dict = None,
                 apply_rotation_trick: bool = False,
                 num_branches: int = 1,
                 **kwargs) -> None:

        super().__init__()

        print(f"Applying rotation trick: {apply_rotation_trick}")
        print(f"Number of branches: {num_branches}")

        self.code_dim = code_dim

        self.encoder = Encoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.decoder = Decoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)
        elif quantizer == "orig":
            self.quantizer = Quantizer(code_num, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(code_num, code_dim, mu=0.99)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(code_num, code_dim)
        
        self.seen_tokens = set()

        self.apply_rotation_trick = apply_rotation_trick

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor):
        # Preprocess
        x_in = self.preprocess(features)

        # Encode
        x_encoder = self.encoder(x_in)

        # quantization
        x_quantized, commit_loss, perplexity = self.quantizer(x_encoder)

        # Compute rotation matrix with detached gradients and apply rotation 
        if self.apply_rotation_trick:
            with torch.no_grad():
                # Normalize vectors for computing rotation
                e_norm = F.normalize(x_encoder.detach(), dim=-1)
                q_norm = F.normalize(x_quantized.detach(), dim=-1)
                
                # Compute r = (e + q)/||e + q|| for Householder reflection
                r = (e_norm + q_norm)
                r = F.normalize(r, dim=-1)
                
                # Compute rotation matrix R = I - 2rr^T + 2qe^T
                B, L, D = x_encoder.shape
                I = torch.eye(D, device=x_encoder.device).expand(B, L, D, D)
                rrt = torch.einsum('bli,blj->blij', r, r)
                qet = torch.einsum('bli,blj->blij', q_norm, e_norm)
                R = I - 2 * rrt + 2 * qet

                # Scale factor to preserve norms
                scaling = (x_quantized.norm(dim=-1) / x_encoder.norm(dim=-1)).unsqueeze(-1)

            # Apply rotation and scaling as constants during backprop
                x_quantized_rotated = scaling * torch.einsum('blij,blj->bli', R, x_encoder)
        else:
            x_quantized_rotated = x_quantized
        
        # decoder
        x_decoder = self.decoder(x_quantized_rotated)
        x_out = self.postprocess(x_decoder)

        return x_out, commit_loss, perplexity
    
    def compute_rotation_matrix(self, e, q):
        """
        Compute rotation matrix using Householder reflections to align z_e with z_q.
        Args:
            z_e: Encoder output (batch_size, seq_len, embedding_dim)
            e: Nearest codebook vector (batch_size, seq_len, embedding_dim)
        Returns:
            Rotation matrices (batch_size, seq_len, embedding_dim, embedding_dim)
        """
        # Normalize vectors
        scaling_factor = (q.norm(dim=-1) / (e.norm(dim=-1) + 1e-6)).unsqueeze(-1)
        
        # Compute r = (e + q)/||e + q||
        r = (e + q) / (e + q).norm(dim=-1, keepdim=True)
        
        # Compute rotation matrix R = I - 2rr^T + 2qe^T
        B, L, D = e.shape
        I = torch.eye(D, device=e.device).expand(B, L, D, D)
        
        # Compute rr^T term
        rrt = torch.einsum('bli,blj->blij', r, r)  # (B, L, D, D)
        
        # Compute qe^T term
        qe_t = torch.einsum('bli,blj->blij', q, e)  # (B, L, D, D)
        
        # Final rotation matrix
        R = I - 2 * rrt + 2 * qe_t  # (B, L, D, D)

        
        return R, scaling_factor

    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:

        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in) # encode to latent space

        x_encoder = self.postprocess(x_encoder) # permutation
        x_encoder = x_encoder.contiguous().view(-1,
                                                x_encoder.shape[-1])  # (NT, C)

        code_idx = self.quantizer.quantize(x_encoder) # quantize to codebook
        code_idx = code_idx.view(N, -1)

        # latent, dist
        return code_idx, None

    def decode(self, z: Tensor):

        x_d = self.quantizer.dequantize(z)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        # Regular stride branch
        blocks1 = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks1.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks1.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate,
                        activation=activation, norm=norm),
            )
            blocks1.append(block)
        blocks1.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.branch1 = nn.Sequential(*blocks1)

        # Wide stride branch
        blocks2 = []
        stride_t_wide = 4  # Wider stride
        filter_t_wide, pad_t_wide = stride_t_wide * 2, stride_t_wide // 2
        blocks2.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks2.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t_wide, stride_t_wide, pad_t_wide),
                Resnet1D(width, depth, dilation_growth_rate,
                        activation=activation, norm=norm),
            )
            blocks2.append(block)
        blocks2.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.branch2 = nn.Sequential(*blocks2)

        # Fusion layer
        self.fusion = nn.Conv1d(output_emb_width * 2, output_emb_width, 1, 1, 0)

    def forward(self, x):
        # Process both branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        
        # Interpolate wider stride branch to match size of regular branch
        out2 = F.interpolate(out2, size=out1.shape[-1], mode='linear', align_corners=False)
        
        # Combine features
        combined = torch.cat([out1, out2], dim=1)
        return self.fusion(combined)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        # Split input into two branches
        self.split = nn.Conv1d(output_emb_width, output_emb_width * 2, 1, 1, 0)

        # Regular stride branch
        blocks1 = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks1.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks1.append(nn.ReLU())
        
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate,
                        reverse_dilation=True, activation=activation,
                        norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks1.append(block)
        
        blocks1.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks1.append(nn.ReLU())
        blocks1.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.branch1 = nn.Sequential(*blocks1)

        # Wide stride branch
        blocks2 = []
        stride_t_wide = 4  # Matching encoder's wider stride
        filter_t_wide, pad_t_wide = stride_t_wide * 2, stride_t_wide // 2
        blocks2.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks2.append(nn.ReLU())
        
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate,
                        reverse_dilation=True, activation=activation,
                        norm=norm),
                nn.Upsample(scale_factor=4, mode='nearest'),  # Wider upsampling
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks2.append(block)
        
        blocks2.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks2.append(nn.ReLU())
        blocks2.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.branch2 = nn.Sequential(*blocks2)

        # Final fusion layer
        self.fusion = nn.Conv1d(input_emb_width * 2, input_emb_width, 1, 1, 0)

    def forward(self, x):
        # Split features for two branches
        split_features = self.split(x)
        features1, features2 = torch.chunk(split_features, 2, dim=1)
        
        # Process both branches
        out1 = self.branch1(features1)
        out2 = self.branch2(features2)
        
        # Interpolate the second branch to match the first branch's size
        out2 = F.interpolate(out2, size=out1.shape[-1], mode='linear', align_corners=False)
        
        # Combine and fuse
        combined = torch.cat([out1, out2], dim=1)
        return self.fusion(combined)
