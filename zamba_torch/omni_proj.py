import torch
import torch.nn as nn


class OmniProj(nn.Module):
    """
    A PyTorch module that applies a linear projection to a specified dimension of a tensor.

    Args:
        input_dim (int): The size of the input dimension to project from.
        output_dim (int): The size of the output dimension to project to.
        dim (int): The dimension of the tensor to apply the projection to.

    Example:
        >>> x = torch.randn(10, 20, 30)
        >>> proj = OmniProj(20, 50, dim=1)
        >>> y = proj(x)
        >>> y.shape
        torch.Size([10, 50, 30])
    """

    def __init__(self, input_dim: int, output_dim: int, dim: int):
        super(OmniProj, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim = dim
        self.linear = nn.Linear(input_dim, output_dim)

        if dim < 0:
            raise ValueError("dim must be a non-negative integer")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear projection.

        Args:
            x (torch.Tensor): The input tensor to project.

        Returns:
            torch.Tensor: The projected tensor.
        """
        # Ensure the dimension to project exists
        if self.dim >= x.ndim:
            raise ValueError(
                f"dim must be less than the number of dimensions of the input tensor, got {self.dim} for tensor with shape {x.shape}"
            )

        # Move the specified dimension to the last position
        permute_dims = list(range(x.ndim))
        permute_dims.append(permute_dims.pop(self.dim))
        x = x.permute(*permute_dims)

        # Apply the linear transformation
        x = self.linear(x)

        # Restore the original dimension order
        inverse_permute_dims = list(range(x.ndim))
        inverse_permute_dims.insert(
            self.dim, inverse_permute_dims.pop(-1)
        )
        x = x.permute(*inverse_permute_dims)

        return x


# # Example usage
# if __name__ == "__main__":
#     x = torch.randn(10, 100, 30)
#     proj = OmniProj(100, 1, dim=1)
#     y = proj(x)
#     print(y.shape)  # Expected shape: torch.Size([10, 50, 30])
