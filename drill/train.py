import os
import torch
import wandb
from config import SAEConfig
from model import SparseAutoencoder
from buffer import ActivationBuffer
from transformer_lens import HookedTransformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SAEConfig(device=device)
    
    # Initialize wandb
    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=cfg.to_dict())

    model = HookedTransformer.from_pretrained(cfg.model_name)
    sae = SparseAutoencoder(cfg)
    sae.to(cfg.device)
    buffer = ActivationBuffer(cfg, model)

    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_steps = cfg.n_training_tokens // cfg.train_batch_size
    steps_since_last_activation = torch.zeros(cfg.d_sae, dtype=torch.int64)

    for step in range(num_steps):
        optimizer.zero_grad()
        acts = buffer.get_activations()
        sae_out, feature_acts, loss, mse_loss, l1_loss = sae(acts)
        loss.backward()
        optimizer.step()
        # print('feature acts shape', feature_acts.shape) # [batch_size, d_sae]

        # Update dead feature tracker
        activated_features = (feature_acts > 0).any(dim=0).cpu()
        steps_since_last_activation[activated_features] = 0
        steps_since_last_activation[~activated_features] += cfg.train_batch_size

        if cfg.log_to_wandb and (step + 1) % cfg.wandb_log_frequency == 0:
            dead_features_prop = (steps_since_last_activation >= cfg.dead_feature_threshold).float().mean()

            wandb.log({
                "step": step,
                "loss": loss.item(),
                "mse_loss": mse_loss.item(),
                "l1_loss": l1_loss.item(),
                "dead_features_prop": dead_features_prop.item(),
            })

        print(f"Step: {step}, Loss: {loss.item()}")

        # Save checkpoint every cfg.checkpoint_frequency steps
        if step % cfg.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
            torch.save(sae.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at step {step}")

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(sae.state_dict(), final_model_path)
    print("Final model saved")

    if cfg.log_to_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
