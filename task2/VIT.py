import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, P, P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class Attention(nn.Module):
    """Attention mechanism."""
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
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

class MLP(nn.Module):
    """Multilayer perceptron within a Transformer block."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 使用int确保参数类型正确
        hidden_features = int(hidden_features if hidden_features is not None else in_features * 4)
        out_features = int(out_features if out_features is not None else in_features)
        print("Initializing MLP:", in_features, hidden_features, out_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)




    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """A single block of a Transformer."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_features=dim * mlp_ratio, out_features=dim)

    def forward(self, x):
        y = self.norm1(x)
        y = self.attn(y)
        x = x + y
        y = self.norm2(x)
        y = self.mlp(y)
        x = x + y
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer model."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100, embed_dim=768, depth=12, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

def ViT():
    return VisionTransformer(
        img_size=32,        # 图像尺寸
        patch_size=4,       # 每个图像块的尺寸
        in_chans=3,         # 输入通道数
        num_classes=100,    # 分类的类别数
        embed_dim=192,      # 减少嵌入维度到192
        depth=4,            # 减少层数到4
        num_heads=4         # 减少头的数量到4
    )

