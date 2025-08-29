import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_
from configs.neuron_factory import create_neuron


class PatchEmbedInit(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=256,
        neuron_type="LIF",
        surrogate_function="sigmoid",
        neuron_args=None,
    ):
        super().__init__()

        # Downsampling + Res 0
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_maxpool = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.proj_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.proj1_conv = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj1_bn = nn.BatchNorm2d(embed_dims)
        self.proj1_maxpool = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.proj1_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.proj2_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj2_bn = nn.BatchNorm2d(embed_dims)
        self.proj2_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.proj_res_conv = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False
        )
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x)
        x = self.proj_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x_feat = x
        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.proj1_maxpool(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()

        x = self.proj2_conv(x)
        x = self.proj2_bn(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.proj2_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat  # shortcut

        return x


class TokenQKAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        neuron_type="LIF",
        surrogate_function="sigmoid",
        neuron_args=None,
        dropout=0.2,
    ):
        super().__init__()

        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.attn_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        # Add dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        T, B, C, H, W = x.shape
        dtype = x.dtype

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(
            T, B, self.num_heads, C // self.num_heads, N
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(
            T, B, self.num_heads, C // self.num_heads, N
        )

        q = torch.sum(q, dim=3, keepdim=True)
        attn = self.attn_lif(q)

        attn = self.attn_dropout(attn.to(dtype))  # Dropout after attention

        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x)
        x = self.proj_dropout(x.to(dtype))  # Dropout after projection

        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        neuron_type="LIF",
        surrogate_function="sigmoid",
        neuron_args=None,
        dropout=0.2,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.fc2_conv = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.c_hidden = hidden_features
        self.c_output = out_features

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T, B, C, W, H = x.shape
        dtype = x.dtype
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, W, H).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, W, H).contiguous()
        x = self.fc2_lif(x)

        x = self.dropout(x.to(dtype))  # Dropout after the second LIF

        return x


class TokenSpikingTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        neuron_type="LIF",
        surrogate_function="sigmoid",
        neuron_args=None,
        dropout=0.2,
    ):
        super().__init__()
        self.tssa = TokenQKAttention(
            dim,
            num_heads,
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
            neuron_args=neuron_args,
            dropout=dropout,
        )
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
            neuron_args=neuron_args,
            dropout=dropout,
        )

    def forward(self, x):
        x = x + self.tssa(x)
        x = x + self.mlp(x)

        return x


class PatchEmbeddingStage(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=512,
        neuron_type="LIF",
        surrogate_function="sigmoid",
        neuron_args=None,
    ):
        super().__init__()
        self.C = in_channels

        self.proj3_conv = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_maxpool = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.proj3_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.proj4_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.proj_res_conv = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False
        )
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.proj3_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat  # shortcut

        return x


class SpikingSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        neuron_type="LIF",
        surrogate_function="sigmoid",
        neuron_args=None,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )
        self.attn_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = create_neuron(
            neuron_type=neuron_type, surrogate_type=surrogate_function, **neuron_args
        )

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = (
            q_conv_out.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = (
            k_conv_out.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = (
            v_conv_out.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, W, H)

        return x


class SpikingTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        neuron_type="LIF",
        surrogate_function="sigmoid",
        neuron_args=None,
    ):
        super().__init__()
        self.attn = SpikingSelfAttention(
            dim,
            num_heads=num_heads,
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
            neuron_args=neuron_args,
        )
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
            neuron_args=neuron_args,
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class HierarchicalSpikingTransformer(nn.Module):
    def __init__(
        self,
        T=4,
        in_channels=3,
        num_classes=10,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        depths=None,
        is_video=True,
        neuron_type="LIF",
        surrogate_function="sigmoid",
        neuron_args=None,
        dropout=0.2,
    ):
        super().__init__()
        if depths is None:
            depths = [1, 2, 3]
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        self.is_video = is_video
        self.patch_embed1 = PatchEmbedInit(
            in_channels=in_channels,
            embed_dims=embed_dims // 4,
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
            neuron_args=neuron_args,
        )
        self.stage1 = nn.ModuleList(
            [
                TokenSpikingTransformer(
                    dim=embed_dims // 4,
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios,
                    neuron_type=neuron_type,
                    surrogate_function=surrogate_function,
                    neuron_args=neuron_args,
                    dropout=dropout,
                )
                for _ in range(depths[0])
            ]
        )

        self.patch_embed2 = PatchEmbeddingStage(
            in_channels=in_channels,
            embed_dims=embed_dims // 2,
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
            neuron_args=neuron_args,
        )
        self.stage2 = nn.ModuleList(
            [
                TokenSpikingTransformer(
                    dim=embed_dims // 2,
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios,
                    neuron_type=neuron_type,
                    surrogate_function=surrogate_function,
                    neuron_args=neuron_args,
                    dropout=dropout,
                )
                for _ in range(depths[1])
            ]
        )

        self.patch_embed3 = PatchEmbeddingStage(
            in_channels=in_channels,
            embed_dims=embed_dims,
            neuron_type=neuron_type,
            surrogate_function=surrogate_function,
            neuron_args=neuron_args,
        )
        self.stage3 = nn.ModuleList(
            [
                SpikingTransformer(
                    dim=embed_dims,
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios,
                    neuron_type=neuron_type,
                    surrogate_function=surrogate_function,
                    neuron_args=neuron_args,
                )
                for _ in range(depths[2])
            ]
        )

        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        x = self.patch_embed1(x)
        for blk in self.stage1:
            x = blk(x)

        x = self.patch_embed2(x)
        for blk in self.stage2:
            x = blk(x)

        x = self.patch_embed3(x)
        for blk in self.stage3:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        if self.is_video:
            # rearranging dims
            x = x.permute(1, 0, 2, 3, 4)  # T, B, C, H, W
        else:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)

        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x