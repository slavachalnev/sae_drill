# like buffer except we just precompute the activations and store on disk

import os
import numpy as np
from tqdm import tqdm
import torch
from transformer_lens import HookedTransformer
from config import SAEConfig
from buffer import ActivationBuffer


def compute_and_save_activations(cfg: SAEConfig, output_file: str, max_batches: int):
    model = HookedTransformer.from_pretrained(cfg.model_name) 
    buffer = ActivationBuffer(cfg, model)
    
    # TODO: replace with train_batch_size when I fix buffer.get_activations
    total_rows = max_batches * cfg.store_batch_size * cfg.context_size
    
    mmap = np.memmap(output_file, dtype=np.float16, mode='w+', shape=(total_rows, cfg.d_in))
    
    row_idx = 0
    for i in tqdm(range(max_batches)):
        activations = buffer.get_activations()
        activations = activations.cpu().numpy().astype(np.float16)
        mmap[row_idx : row_idx + activations.shape[0]] = activations
        row_idx += activations.shape[0]


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    dir = "/mnt/hdd/activation_cache/NeelNanda/c4-code-tokenized-2b/gelu-2l"
    os.makedirs(dir, exist_ok=True)
    cfg = SAEConfig(
        device="cuda",
        store_batch_size=32,
        n_batches_in_buffer=100,
    )
    # ctx len is 1024
    compute_and_save_activations(cfg, f"{dir}/activations_5k.npy", max_batches=500) # 60000
