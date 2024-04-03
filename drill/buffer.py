import os
from typing import Any, Iterator, cast

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer


class ActivationBuffer:
    def __init__(self, cfg: Any, model: HookedTransformer):
        self.cfg = cfg
        self.model = model
        self.dataset = load_dataset(cfg.dataset_path, split="train", streaming=True)
        self.token_loader = DataLoader(self.dataset, batch_size=cfg.store_batch_size, shuffle=True)

        self.buffer = []
        self.buffer_size = cfg.n_batches_in_buffer * cfg.store_batch_size
        self.batch_idx = 0
    
    def fill_buffer(self):
        while len(self.buffer) < self.buffer_size:
            tokens = self.get_token_batch()
            activations = self.model.run_with_cache(tokens, stop_at_layer=self.cfg.hook_point_layer + 1)[1][self.cfg.hook_point]
            self.buffer.append(activations)
        # shuffle
        self.buffer = self.buffer[torch.randperm(len(self.buffer))]
        self.batch_idx = 0

    def get_activations(self):
        if self.idx > self.cfg.n_batches_in_buffer // 2:
            self.buffer = self.buffer[self.batch_idx:]
            self.fill_buffer()
        activations = torch.stack(self.buffer[:self.cfg.store_batch_size], dim=0)
        self.batch_idx += 1
        return activations

    def get_token_batch(self):
        try:
            return next(self.token_loader)
        except StopIteration:
            self.token_loader = DataLoader(self.dataset, batch_size=self.cfg.store_batch_size, shuffle=True)
            return next(self.token_loader)

