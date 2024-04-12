# like buffer except we just precompute the activations and store on disk

import os
import numpy as np
from tqdm import tqdm
import torch
from transformer_lens import HookedTransformer
from config import SAEConfig
from datasets import load_dataset
from torch.utils.data import DataLoader


def compute_and_save_activations(cfg: SAEConfig, output_file: str, max_batches=None):
    batch_size = 16
    model = HookedTransformer.from_pretrained(cfg.model_name) 
    dataset = load_dataset(cfg.dataset_path, split="train", streaming=True)
    dataset = dataset.with_format("torch")
    token_loader = iter(DataLoader(dataset, batch_size=batch_size))
    hook_point = cfg.hook_point.format(layer=cfg.hook_point_layer)
    
    if max_batches is None:
        max_batches = len(dataset) // batch_size
    total_rows = max_batches * batch_size * cfg.context_size
    
    mmap = np.memmap(output_file, dtype=np.float16, mode='w+', shape=(total_rows, cfg.d_in))
    
    row_idx = 0
    for i in tqdm(range(max_batches)):
        tokens = next(token_loader)['tokens']
        activations = model.run_with_cache(tokens, stop_at_layer=cfg.hook_point_layer + 1)[1][hook_point]
        activations = activations.view(-1, cfg.d_in)
        activations = activations.cpu().numpy().astype(np.float16)
        mmap[row_idx : row_idx + activations.shape[0]] = activations
        row_idx += activations.shape[0]


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    dir = "/mnt/hdd/activation_cache/NeelNanda/c4-code-tokenized-2b/gelu-2l"
    os.makedirs(dir, exist_ok=True)
    cfg = SAEConfig(device="cuda")
    compute_and_save_activations(cfg, f"{dir}/activations_5k.npy", max_batches=5000) # 60000

