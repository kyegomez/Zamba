import torch  # Importing the torch library for deep learning operations
from zamba_torch.main import (
    ZambaBlock,
)  # Importing the ZambaBlock class from the zamba.main module


x = torch.randn(
    1, 512, 512
)  # Generating a random tensor of shape (1, 512, 512)
model = ZambaBlock(
    dim=512,  # Setting the dimension of the model to 512
    heads=8,  # Setting the number of attention heads to 8
    dim_head=64,  # Setting the dimension of each attention head to 64
    d_state=512,  # Setting the state dimension to 512
    dt_rank=128,  # Setting the rank of the temporal kernel to 128
    d_conv=256,  # Setting the dimension of the convolutional layer to 256
)
print(
    model(x)
)  # Printing the output of the model when applied to the input tensor
