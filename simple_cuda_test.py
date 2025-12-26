import sys
import torch
print(f"Executable: {sys.executable}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Torch Version: {torch.__version__}")
