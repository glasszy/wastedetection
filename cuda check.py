import torch
import torchvision
print(torch.__version__)       # should be 2.7.1+cu126
print(torchvision.__version__)  # should be compatible and not CPU-only
print(torch.cuda.is_available())  # should be True