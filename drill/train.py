import torch

from config import SAEConfig
from model import SparseAutoencoder
from buffer import ActivationBuffer

from transformer_lens import HookedTransformer



def main():
    cfg = SAEConfig()
    model = HookedTransformer.from_pretrained(cfg.model_name)
    sae = SparseAutoencoder(cfg)
    buffer = ActivationBuffer(cfg, model)

    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    for i in range(10):
        optimizer.zero_grad()
        acts = buffer.get_activations()
        sae_out, feature_acts, loss, mse_loss, l1_loss = sae(acts)
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

if __name__ == "__main__":
    main()
