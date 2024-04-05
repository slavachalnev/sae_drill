import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from config import SAEConfig


class ActivationBuffer:
    def __init__(self, cfg: SAEConfig, model: HookedTransformer):
        self.cfg: SAEConfig = cfg
        self.model: HookedTransformer = model
        self.dataset = load_dataset(cfg.dataset_path, split="train", streaming=True)
        self.dataset = self.dataset.with_format("torch")
        self.token_loader = iter(DataLoader(self.dataset, batch_size=cfg.store_batch_size))
        self.hook_point = self.cfg.hook_point.format(layer=self.cfg.hook_point_layer)
        self.buffer_size = cfg.n_batches_in_buffer * cfg.store_batch_size * cfg.context_size
        self.buffer = torch.zeros((self.buffer_size, self.cfg.d_in), dtype=torch.float32, device=cfg.device)
        self.batch_idx = 0
        self.fill_buffer()

    def fill_buffer(self):
        buffer_index = self.batch_idx * self.cfg.store_batch_size * self.cfg.context_size
        while buffer_index < self.buffer_size:
            tokens = self.get_token_batch()
            activations = self.model.run_with_cache(tokens, stop_at_layer=self.cfg.hook_point_layer + 1)[1][self.hook_point]
            activations = activations.view(-1, self.cfg.d_in)
            self.buffer[buffer_index : buffer_index + self.cfg.store_batch_size * self.cfg.context_size] = activations
            buffer_index += self.cfg.store_batch_size * self.cfg.context_size
        self.buffer = self.buffer[torch.randperm(self.buffer_size)]
        self.batch_idx = 0

    def get_activations(self):
        if self.batch_idx > self.cfg.n_batches_in_buffer // 2:
            # self.buffer = self.buffer[self.batch_idx * self.cfg.store_batch_size :]
            self.fill_buffer()
        from_idx = self.batch_idx * self.cfg.store_batch_size * self.cfg.context_size
        to_idx = (self.batch_idx + 1) * self.cfg.store_batch_size * self.cfg.context_size
        activations = self.buffer[from_idx : to_idx]
        self.batch_idx += 1
        return activations

    def get_token_batch(self):
        try:
            return next(self.token_loader)['tokens']
        except StopIteration:
            self.token_loader = iter(DataLoader(self.dataset, batch_size=self.cfg.store_batch_size))
            return next(self.token_loader)['tokens']
