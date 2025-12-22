# cli.py
import torch
from lightning.pytorch.cli import LightningCLI

# enable tensor cores for RTX 5090
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

def main():
    LightningCLI(
        save_config_kwargs={"overwrite": True},
        subclass_mode_model=True,
        subclass_mode_data=True,
        ) 
    
if __name__ == "__main__":
    main()
