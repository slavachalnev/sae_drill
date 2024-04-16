# like buffer except we just precompute the activations and store on disk

import os
import json
import numpy as np
from tqdm import tqdm
import torch
from transformer_lens import HookedTransformer
from config import SAEConfig
from buffer import ActivationBuffer
from model import SparseAutoencoder


def compute_and_save_activations(cfg: SAEConfig, output_file: str, max_batches: int, filter=None):
    model = HookedTransformer.from_pretrained(cfg.model_name) 
    buffer = ActivationBuffer(cfg, model, filter=filter)
    
    total_rows = max_batches * cfg.train_batch_size
    
    mmap = np.memmap(output_file, dtype=np.float16, mode='w+', shape=(total_rows, cfg.d_in))
    
    row_idx = 0
    for i in tqdm(range(max_batches)):
        activations = buffer.get_activations()
        activations = activations.cpu().numpy().astype(np.float16)
        mmap[row_idx : row_idx + activations.shape[0]] = activations
        row_idx += activations.shape[0]


# if __name__ == "__main__":
#     torch.set_grad_enabled(False)

#     dir = "/mnt/hdd/activation_cache/NeelNanda/c4-code-tokenized-2b/gelu-2l"
#     os.makedirs(dir, exist_ok=True)
#     cfg = SAEConfig(
#         device="cuda",
#         store_batch_size=32,
#         n_batches_in_buffer=100,
#         train_batch_size=32*1024,
#     )

#     # ctx len is 1024. so 32 * 1024 * 60000 is approx 2B tokens.
#     compute_and_save_activations(cfg, f"{dir}/activations_5k.npy", max_batches=5000) # 60000


def get_filter(sae_checkpoint_dir: str, feature_id: int):
    with open(os.path.join(sae_checkpoint_dir, "config.json")) as f:
        sae_cfg = json.load(f)
    sae_cfg = SAEConfig(**sae_cfg)
    sae_cfg.device = "cuda"
    sae = SparseAutoencoder(sae_cfg)

    def feature_filter(x):
        # x: (batch_size, d_in)
        x = x.to(sae_cfg.device)
        x = x - sae.b_dec
        x = x @ sae.W_enc[:, feature_id]
        x += sae.b_enc[feature_id]
        x = torch.nn.functional.relu(x)
        active = x > 0
        # print('num active', active.sum().item(), 'out of', active.shape[0])
        return active
    
    return feature_filter


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sae_dir = "checkpoints/2024-04-13_12-30-52" # expansion factor = 8

    feature_id = 3
    dir = "/mnt/hdd/activation_cache/NeelNanda/c4-code-tokenized-2b/gelu-2l"
    os.makedirs(dir, exist_ok=True)
    cfg = SAEConfig(
        device="cuda",
        store_batch_size=32,
        n_batches_in_buffer=100,
        train_batch_size=32*1024,
    )

    # ctx len is 1024. so 32 * 1024 * 60000 is approx 2B tokens.
    compute_and_save_activations(cfg,
                                 f"{dir}/acts_2k_ft_{feature_id}.npy",
                                 max_batches=2000, # 60000
                                 filter=get_filter(sae_dir, feature_id)
                                 )
