import os
import requests
from tqdm import tqdm
import torch

CHUNK_SIZE = 8192

MODEL_CACHE_DIR = os.path.expanduser("~/.chet/models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def _ensure_model_downloaded(name: str, url: str):
    model_path = os.path.join(MODEL_CACHE_DIR, name)
    if not os.path.exists(model_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(model_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))

    return model_path

def chet_1_base(device: str = "cpu"):
    from .model import ModelConfig, Chet

    config = ModelConfig(
        embed_dim=480,
        n_heads=12,
        n_layers=12,
        dropout=0.1,
    )
    model = Chet(config)

    path = _ensure_model_downloaded("chet_1_base.pt", "https://s3.ianhuang.dev/models/chet_1_base.pt")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)