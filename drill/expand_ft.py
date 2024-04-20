# take a model, sae, and select feature.
# Split the feature into 8 features.
import wandb
import time
import os
import json
import torch
import einops
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
    for param in sae.parameters():
        param.requires_grad = False
    sae.load_state_dict(torch.load(os.path.join(checkpoint_dir, "final_model.pt"), map_location=torch.device(sae_cfg.device)))
    sae.to(sae_cfg.device)

    # copy the feature and delete it from the model.
    feature_enc_dec = torch.stack([sae.W_enc[:, feature_id], sae.W_dec[feature_id]])
    sae.W_enc[:, feature_id] = 0
    sae.W_dec[feature_id] = 0

    @torch.no_grad()
    def feature_filter(x):
        # x: (batch_size, d_in)
        x = x.to(sae_cfg.device)
        x = x - sae.b_dec
        x = x @ feature_enc_dec[0]
        x += sae.b_enc[feature_id]
        x = torch.nn.functional.relu(x)
        return x > 0

    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        drill_cfg = json.load(f)
    drill_cfg = SAEConfig(**drill_cfg)
    drill_cfg.device = device
    drill_cfg.from_pretrained_path = checkpoint_dir
    drill_cfg.d_sae = 8  # split original feature into 8 features
    drill_cfg.lr = 1e-4
    drill_cfg.noise_scale = 0.02
    drill_cfg.wandb_log_frequency = 100
    drill_cfg.n_training_tokens = int(5e8)

    if drill_cfg.log_to_wandb:
        wandb.init(project="drill", name=drill_cfg.run_name, config=drill_cfg.to_dict())

    drill = DrillSAE(drill_cfg, feature_enc_dec)
    drill.to(drill_cfg.device)

    # model = HookedTransformer.from_pretrained(sae_cfg.model_name, device=sae_cfg.device)
    # buffer = ActivationBuffer(drill_cfg, model, filter=feature_filter)
    buffer = ActivationLoader(
        np_path="/home/slava/activation_cache/NeelNanda/c4-code-tokenized-2b/gelu-2l/acts_1k_ft_3.npy",
        cfg=drill_cfg,
    )

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

        drill_out, feature_acts, l1_loss = drill(x)

        sum_out = sae_out + drill_out
        x_centred = x - x.mean(dim=0, keepdim=True)
        mse_loss = torch.pow((sum_out - x.float()), 2)/ (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
        mse_loss = mse_loss.mean()
        loss = mse_loss + l1_loss

        loss.backward()
        drill.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        drill.set_decoder_norm_to_unit_norm()

        l_0 = (feature_acts > 0).float().sum(dim=-1).mean()
        if drill_cfg.log_to_wandb and (step + 1) % drill_cfg.wandb_log_frequency == 0:
            wandb.log({
                "step": step,
                "loss": loss.item(),
                "mse_loss": mse_loss.item(),
                "l1_loss": l1_loss.item(),
                "l_0": l_0.item(),
            })

        # TODO: compute activation freq for every feature in drill.

        if step % 10 == 0:
            print(f"Step: {step}, Loss: {loss.item()}, MSE Loss: {mse_loss.item()}, L1 Loss: {l1_loss.item()}")
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(sae.state_dict(), final_model_path)
    
    if drill_cfg.log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
