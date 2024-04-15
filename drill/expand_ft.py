# take a model, sae, and select feature.
# Split the feature into 8 features.
import wandb
import time
import os
import json
import torch
from config import SAEConfig
from model import SparseAutoencoder, DrillSAE
from buffer import ActivationLoader, ActivationBuffer
from transformer_lens import HookedTransformer

def main():
    feature_id = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = "checkpoints/2024-04-13_12-30-52"  # expansion factor = 8

    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        sae_cfg = json.load(f)
    sae_cfg = SAEConfig(**sae_cfg)
    sae_cfg.device = device

    sae = SparseAutoencoder(sae_cfg)
    sae.load_state_dict(torch.load(os.path.join(checkpoint_dir, "final_model.pt"), map_location=torch.device(sae_cfg.device)))
    sae.to(sae_cfg.device)

    # copy the feature and delete it from the model.
    feature_enc_dec = torch.stack([sae.W_enc[:, feature_id], sae.W_dec[feature_id]])
    sae.W_enc[:, feature_id] = 0
    sae.W_dec[feature_id] = 0

    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        drill_cfg = json.load(f)
    drill_cfg = SAEConfig(**drill_cfg)
    drill_cfg.device = device
    drill_cfg.from_pretrained_path = checkpoint_dir
    drill_cfg.d_sae = 8  # split original feature into 8 features
    drill_cfg.lr = 1e-4
    drill_cfg.noise_scale = 0.02

    drill = DrillSAE(drill_cfg, feature_enc_dec)
    drill.to(drill_cfg.device)

    model = HookedTransformer.from_pretrained(sae_cfg.model_name, device=sae_cfg.device)

    # TODO: pass sae and feature to buffer for filtering.
    buffer = ActivationBuffer(drill_cfg, model)

    optimizer = torch.optim.Adam(drill.parameters(), lr=drill_cfg.lr)

    checkpoint_dir = f"d_ckpts/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(drill_cfg.to_dict(), f, indent=2)
    
    num_steps = drill_cfg.n_training_tokens // drill_cfg.train_batch_size

    for step in range(num_steps):
        optimizer.zero_grad()
        x = buffer.get_activations()
        x = x.to(drill_cfg.device)

        with torch.no_grad():
            sae_out, _, _, _, _ = sae(x)

        drill_out, _, l1_loss = drill(x)

        sum_out = sae_out + drill_out
        x_centred = x - x.mean(dim=0, keepdim=True)
        mse_loss = torch.pow((sum_out - x.float()), 2)/ (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
        mse_loss = mse_loss.mean()
        loss = mse_loss + l1_loss

        loss.backward()
        drill.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        drill.set_decoder_norm_to_unit_norm()
    


if __name__ == "__main__":
    main()
