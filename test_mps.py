import torch

print("Torch version:", torch.__version__)
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Built-in:", torch.backends.mps.is_built())

x = torch.ones(3, device="mps")
print("Tensor on MPS:", x)