# Filter-Response-Normalization-PyTorch
Unofficial Pytorch implementation of the paper Filter Response Normalization(https://arxiv.org/abs/1911.09737).

## Example
### Using `convert_model` function
``` python
import torch
from torch import nn
import numpy as np
import torchvision
from filter_response_normalization import convert_model
m = torchvision.models.resnet18(True)
m = nn.DataParallel(m)
m = convert_model(m)
x = np.zeros((3, 3, 384, 576), dtype="f")
x = torch.from_numpy(x)
y = m(x)
print(y.size()) # Output:torch.Size([3, 1000])
```


## Issue
* ~Change input shape of TLU  
    In paper implement, threshold shape of `ThresholdedLinearUnit` is `(1, channels, 1, 1) `  
    But in this implement, `ThresholdedLinearUnit` threshold is **scolor**. So if you do not use the convert_model function, use ThresholdedLinearUnitFixND.~  
It was solved by [qubvel](https://github.com/qubvel) contribution. :tada:

Distributed under **MIT License** (See LICENSE)
