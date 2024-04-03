import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from drill.config import SAEConfig


class ActivationBuffer:
    def __init__(self, cfg: SAEConfig, model: HookedTransformer):
        self.cfg: SAEConfig = cfg
        self.model: HookedTransformer = model
        self.dataset = load_dataset(cfg.dataset_path, split="train", streaming=True)
        self.token_loader = DataLoader(self.dataset, batch_size=cfg.store_batch_size, shuffle=True)
        self.buffer_size = cfg.n_batches_in_buffer * cfg.store_batch_size
        self.buffer = torch.zeros((self.buffer_size, *self.cfg.d_in), dtype=torch.float32)
        self.batch_idx = 0

    def fill_buffer(self):
        buffer_index = self.batch_idx * self.cfg.store_batch_size
        while buffer_index < self.buffer_size:
            tokens = self.get_token_batch()
            activations = self.model.run_with_cache(tokens, stop_at_layer=self.cfg.hook_point_layer + 1)[1][self.cfg.hook_point]
            self.buffer[buffer_index : buffer_index + self.cfg.store_batch_size] = activations
            buffer_index += self.cfg.store_batch_size
        self.buffer = self.buffer[torch.randperm(self.buffer_size)]
        self.batch_idx = 0

    def get_activations(self):
        if self.batch_idx > self.cfg.n_batches_in_buffer // 2:
            self.buffer = self.buffer[self.batch_idx * self.cfg.store_batch_size :]
            self.fill_buffer()
        activations = self.buffer[self.batch_idx * self.cfg.store_batch_size : (self.batch_idx + 1) * self.cfg.store_batch_size]
        self.batch_idx += 1
        return activations

    def get_token_batch(self):
        try:
            return next(self.token_loader)
        except StopIteration:
            self.token_loader = DataLoader(self.dataset, batch_size=self.cfg.store_batch_size, shuffle=True)
            return next(self.token_loader)
