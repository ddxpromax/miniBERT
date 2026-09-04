import platform
import sys

import numpy as np
import torch


def main() -> None:
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Numpy: {np.__version__}")
    print(f"CUDA availabe: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    x = torch.randn(1024, 1024, device=device)
    y = x @ x
    print(f"Smoke-test device: {y.device}")
    print(f"Smoke-test result finite: {torch.isfinite(y).all().item()}")

if __name__ == "__main__":
    main()