import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# Model definition (ViT-Tiny for CIFAR-10)
# -----------------------------

class GNLikeLayerNorm(nn.Module):
    """
    GroupNorm(1, D) applied over the last dimension, behaving similar to LayerNorm
    for tensors of shape (B, L, D).
    """
    def __init__(self, normalized_shape, eps=1e-5, affine=True):
        super().__init__()
        self.dim = normalized_shape
        self.norm = nn.GroupNorm(1, self.dim, eps=eps, affine=affine)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        assert D == self.dim, f"Expected last dim={self.dim}, got {D}"

        # reshape so that D is channels
        x = x.reshape(B * L, D, 1)  # (B*L, C=D, 1)
        x = self.norm(x)            # GroupNorm over channels
        x = x.view(B, L, D)         # back to (B, L, D)
        return x
    
class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Input:  (B, C, H, W)
    Output: (B, N_patches, embed_dim)
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Conv2d with kernel_size=stride=patch_size acts as patchify + linear projection
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        if (H, W) != self.img_size:
            raise ValueError(
                f"Expected input size {(self.img_size[0], self.img_size[1])}, "
                f"but got {(H, W)}."
            )

        # B, C, H, W -> B, embed_dim, H', W'
        x = self.proj(x)
        # Flatten spatial dims: B, embed_dim, H', W' -> B, embed_dim, N
        x = x.flatten(2)
        # B, embed_dim, N -> B, N, embed_dim
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    """
    Multi-head Self-Attention.
    """
    def __init__(self, dim, num_heads=3, qkv_bias=True, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    """
    Feed-forward MLP in Transformer block.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    ViT-style Transformer encoder block:
      LN -> MHSA -> residual
      LN -> MLP  -> residual
    """
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attn_dropout=0.0
    ):
        super().__init__()
        self.norm1 = GNLikeLayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.norm2 = GNLikeLayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTinyCIFAR10(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches  # should be 64 for 32x32 / 4x4

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(depth)
        ])

        self.norm = GNLikeLayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        # B, C, H, W -> B, N, D
        x = self.patch_embed(x)

        # prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 1 + N, D)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        x = self.head(cls_out)
        return x
