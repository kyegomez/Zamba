[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Zamba
Implementation of Zamba, the joint mamba-transformer model!, It's now fully ready to train! [PAPER LINK](https://arxiv.org/abs/2405.16712)

# Install
`pip3 install zamba-torch`

## Usage
```python
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
```

# License
MIT

## Citation
```bibtex
@misc{glorioso2024zamba,
    title={Zamba: A Compact 7B SSM Hybrid Model}, 
    author={Paolo Glorioso and Quentin Anthony and Yury Tokpanov and James Whittington and Jonathan Pilault and Adam Ibrahim and Beren Millidge},
    year={2024},
    eprint={2405.16712},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```