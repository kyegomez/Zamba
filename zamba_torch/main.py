import torch
from torch import nn, Tensor
from zeta.nn import Attention, FeedForward
from mambabyte.model import MambaConfig, Mamba
from zamba_torch.omni_proj import OmniProj


class MambaFractralBlock(nn.Module):
    """
    MambaFractralBlock is a PyTorch module that represents a fractal block using the Mamba library.

    Args:
        dim (int): The input dimension.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        d_state (int): The state dimension.
        dt_rank (int): The rank of the tensor train decomposition.
        d_conv (int): The convolutional dimension.
        depth (int): The number of fractal layers.

    Attributes:
        dim (int): The input dimension.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        d_state (int): The state dimension.
        dt_rank (int): The rank of the tensor train decomposition.
        d_conv (int): The convolutional dimension.
        mamba (Mamba): The Mamba instance.
        layers (nn.ModuleList): The list of fractal layers.
        norm (nn.LayerNorm): The layer normalization module.

    """

    def __init__(
        self,
        dim: int = None,
        heads: int = None,
        dim_head: int = None,
        d_state: int = None,
        dt_rank: int = None,
        d_conv: int = None,
        depth: int = 6,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv

        # Mamba config
        mamba_config = MambaConfig(
            dim=dim,
            depth=1,
            dt_rank=dt_rank,
            d_conv=d_conv,
            pscan=True,
        )

        # Mamba
        self.mamba = Mamba(mamba_config)

        # Layers with mamba
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                self.mamba,
            )

        # Norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MambaFractralBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        for layer in self.layers:
            x = layer(x) + x
        return self.norm(x)


# x = torch.randn(1, 512, 512)
# model = MambaFractralBlock(
#     dim=512,
#     heads=8,
#     dim_head=64,
#     d_state=512,
#     dt_rank=128,
#     d_conv=256,
# )
# print(model(x))


class ZambaSharedBlock(nn.Module):
    """
    ZambaSharedBlock is a module that performs attention and feed-forward operations on the input tensor.

    Args:
        dim (int): The dimension of the input tensor.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        d_state (int): The dimension of the state.
        dt_rank (int): The rank of the tensor.
        d_conv (int): The dimension of the convolutional layer.

    Attributes:
        attention (Attention): The attention module.
        norm (nn.LayerNorm): The layer normalization module.
        ffn (FeedForward): The feed-forward module.

    """

    def __init__(
        self,
        dim: int = None,
        heads: int = None,
        dim_head: int = None,
        d_state: int = None,
        dt_rank: int = None,
        d_conv: int = None,
    ):
        super().__init__()

        self.attention = Attention(
            dim,
            dim_head,
            heads,
            qk_norm=True,
            kv_heads=4,
        )

        # Norm
        self.norm = nn.LayerNorm(dim)

        # Mammba
        self.ffn = FeedForward(
            dim,
            dim,
            4,
            swish=True,
            dropout=0.1,
            post_act_ln=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ZambaSharedBlock module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after performing attention and feed-forward operations.

        """
        x, _ = self.attention(x)
        x = self.norm(x)
        x = self.norm(self.ffn(x))
        return x


# x = torch.randn(1, 512, 512)
# model = ZambaSharedBlock(
#     dim=512,
#     heads=8,
#     dim_head=64,
#     d_state=512,
#     dt_rank=128,
#     d_conv=256,
# )
# print(model(x))


class ZambaBlock(nn.Module):
    """
    ZambaBlock is a module that implements a specific block in the Zamba model architecture.

    Args:
        dim (int): The input dimension.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        d_state (int): The state dimension.
        dt_rank (int): The rank of the dynamic tensor.
        d_conv (int): The convolutional dimension.
        depth (int): The depth of the fractal block.

    Attributes:
        dim (int): The input dimension.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        d_state (int): The state dimension.
        dt_rank (int): The rank of the dynamic tensor.
        d_conv (int): The convolutional dimension.
        shared (ZambaSharedBlock): The shared Mamba block.
        fractral (MambaFractralBlock): The fractal Mamba block.
        norm (nn.LayerNorm): The layer normalization module.
        proj (nn.Linear): The linear projection module.
        attention (Attention): The attention module.
        fractral2 (MambaFractralBlock): The second fractal Mamba block.

    """

    def __init__(
        self,
        dim: int = None,
        heads: int = None,
        dim_head: int = None,
        d_state: int = None,
        dt_rank: int = None,
        d_conv: int = None,
        depth: int = 6,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv

        # Shared Mamba
        self.shared = ZambaSharedBlock(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            d_state=d_state,
            dt_rank=dt_rank,
            d_conv=d_conv,
        )

        # Fractral Mamba
        self.fractral = MambaFractralBlock(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            d_state=d_state,
            dt_rank=dt_rank,
            d_conv=d_conv,
            depth=depth,
        )

        # Norms + Projections
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

        # Attention
        self.attention = Attention(
            dim,
            dim_head,
            heads,
            qk_norm=True,
            kv_heads=4,
        )

        # Second fractral
        self.fractral2 = MambaFractralBlock(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            d_state=d_state,
            dt_rank=dt_rank,
            d_conv=d_conv,
            depth=depth,
        )

        #

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Forward pass of the ZambaBlock module.

        Args:
            x (Tensor): The input tensor.
            mask (Tensor, optional): The mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor.

        """
        batch, seq, dim = x.shape
        # device = x.device

        skip = x
        # x = self.norm(self.fractral(x))
        x = self.fractral(x)
        print(x.shape)
        first_fractral = x

        # Concatenated
        concated = torch.cat([skip, x], dim=1)
        b, s, d = concated.shape
        concated = OmniProj(s, seq, dim=1)(concated)

        # Shared
        x = self.shared(concated)

        # Proj
        x = self.proj(x)

        # Second Fractral Mamba
        print(x.shape)
        x = self.fractral2(x + first_fractral)
        # print(x.shape)
        print(f"Second Fractral: {x.shape}")
        second_fractral = x

        # Concatenate
        x = torch.cat([skip, x], dim=1)
        b, s, d = x.shape
        x = OmniProj(s, seq, dim=1)(x)

        # Shared
        x = self.shared(x)

        # Proj
        x = self.proj(x)

        # Third Fractral Mamba
        x = self.fractral(x + second_fractral)

        return self.norm(x)


# x = torch.randn(1, 512, 512)
# model = ZambaBlock(
#     dim=512,
#     heads=8,
#     dim_head=64,
#     d_state=512,
#     dt_rank=128,
#     d_conv=256,
# )
# print(model(x))
