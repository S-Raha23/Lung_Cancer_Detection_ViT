import torch
import numpy as np
from model import CNNTransformer3D_CPU

DEVICE = "cpu"
MODEL_PATH = "cnn_transformer_cpu.pth"

model = CNNTransformer3D_CPU()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict(volume_np):
    """
    Accepts:
    - (H, W)          -> single slice
    - (D, H, W)       -> volume
    - (1, D, H, W)    -> volume with channel

    Converts everything to (1, 1, D, H, W)
    """

    # Case 1: single slice (H, W)
    if volume_np.ndim == 2:
        volume_np = volume_np[np.newaxis, :, :]   # (1, H, W)

    # Case 2: volume (D, H, W)
    if volume_np.ndim == 3:
        volume_np = volume_np[np.newaxis, :, :, :]  # (1, D, H, W)

    # Add channel dim → (1, 1, D, H, W)
    volume_np = volume_np[:, np.newaxis, :, :, :]

    with torch.no_grad():
        x = torch.tensor(volume_np, dtype=torch.float32)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    return prob

