import torch  # Importing the torch library for deep learning operations
from zamba_torch.main import (
    Zamba,
)  # Importing the ZambaBlock class from the zamba.main module

# # Example usage
x = torch.randint(
    0, 256, (1, 512)
)  # Generating a random tensor of shape (1, 512, 512

model = Zamba(
    dim=512,  # Setting the dimension of the model to 512
    heads=8,  # Setting the number of attention heads to 8
    dim_head=64,  # Setting the dimension of each attention head to 64
    d_state=512,  # Setting the state dimension to 512
    dt_rank=128,  # Setting the rank of the temporal kernel to 128
    d_conv=256,  # Setting the dimension of the convolutional layer to 256
    vocab_size=256,  # Setting the size of the vocabulary to 256
    max_seq_len=512,  # Setting the maximum sequence length to 512
)

print(
    model(x)
)  # Printing the output of the model when applied to the input tensor
