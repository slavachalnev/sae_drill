import os
import time
import wandb
import torch
from config import SAEConfig
from model import SparseAutoencoder
from buffer import ActivationBuffer, ActivationLoader
from transformer_lens import HookedTransformer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SAEConfig(device=device,
                    # steps_between_resample=10, ## for testing
                    # log_to_wandb=False,  ## for testing
                    # n_batches_in_buffer=10,  ## for testing
                    checkpoint_frequency=None,
                    )
    
    # Initialize wandb
    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=cfg.to_dict())
    
    sae = SparseAutoencoder(cfg)
    sae.to(cfg.device)

    # model = HookedTransformer.from_pretrained(cfg.model_name)
    # buffer = ActivationBuffer(cfg, model)
    buffer = ActivationLoader(
        np_path="/home/slava/activation_cache/NeelNanda/c4-code-tokenized-2b/gelu-2l/activations_5k.npy",
        cfg=cfg
        )

    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    if cfg.lr_scheduler_name == "constant":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    elif cfg.lr_scheduler_name == "constantwithwarmup":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / cfg.lr_warm_up_steps))
    else:
        raise ValueError(f"Unknown lr_scheduler_name: {cfg.lr_scheduler_name}")

    # Create checkpoint directory
    checkpoint_dir = f"checkpoints/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_steps = cfg.n_training_tokens // cfg.train_batch_size
    steps_since_last_activation = torch.zeros(cfg.d_sae, dtype=torch.int64)

    t = time.time()

    for step in range(num_steps):
        if cfg.l1_warm_up:
            l1_factor = min(1.0, step / 1000)
        else:
            l1_factor = 1.0

        optimizer.zero_grad()
        acts = buffer.get_activations()
        acts = acts.to(cfg.device)

        sae_out, feature_acts, loss, mse_loss, l1_loss = sae(acts, l1_factor=l1_factor)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Update dead feature tracker
        activated_features = (feature_acts > 0).any(dim=0).cpu()
        steps_since_last_activation[activated_features] = 0
        steps_since_last_activation[~activated_features] += cfg.train_batch_size

        if cfg.log_to_wandb and (step + 1) % cfg.wandb_log_frequency == 0:
            dead_features_prop = (steps_since_last_activation >= cfg.dead_feature_threshold).float().mean()

            l_0 = (feature_acts > 0).float().sum(dim=-1).mean()

            wandb.log({
                "step": step,
                "loss": loss.item(),
                "mse_loss": mse_loss.item(),
                "l1_loss": l1_loss.item(),
                "dead_features_prop": dead_features_prop.item(),
                "l_0": l_0.item(),
            })

        # if step % 10 == 0:
        #     print(f"Step: {step}, Loss: {loss.item()}, Time: {time.time() - t}")
        #     t = time.time()

        # Save checkpoint every cfg.checkpoint_frequency steps
        if cfg.checkpoint_frequency and (step + 1) % cfg.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
            torch.save(sae.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at step {step}")
        
        if (step + 1) % cfg.steps_between_resample == 0:
            dead_idxs = steps_since_last_activation >= cfg.dead_feature_threshold
            if dead_idxs.sum() != 0:
                resample(sae=sae, buffer=buffer, dead_idxs=dead_idxs)
                if cfg.tune_resample:
                    tune_resample(sae=sae, buffer=buffer, dead_idxs=dead_idxs)
                reset_optimizer(sae, optimizer, dead_idxs)
                steps_since_last_activation[dead_idxs] = 0

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save(sae.state_dict(), final_model_path)
    print("Final model saved")

    if cfg.log_to_wandb:
        wandb.finish()


def resample(sae: SparseAutoencoder, buffer: ActivationBuffer, dead_idxs):
    # find the inputs where loss is high
    n_resample_steps = 200
    all_acts = []
    all_outs = []
    all_loss = []
    for _ in range(n_resample_steps):
        acts = buffer.get_activations()  # [4096, 512]
        acts = acts.to(sae.W_enc.device)

        with torch.no_grad():
            sae_out = sae(acts)[0]
        
        mse_losses = ((sae_out - acts) ** 2).mean(dim=-1)
        all_acts.append(acts.to('cpu'))
        all_outs.append(sae_out.to('cpu'))
        all_loss.append(mse_losses.to('cpu'))
    
    all_acts = torch.cat(all_acts, dim=0)
    all_outs = torch.cat(all_outs, dim=0)
    all_loss = torch.cat(all_loss, dim=0)

    probs = all_loss ** 2
    probs = probs / probs.sum()

    n_dead = dead_idxs.sum().item()
    resample_idxs = torch.multinomial(probs, n_dead, replacement=True)
    resample_acts = all_acts[resample_idxs]
    resample_acts = resample_acts.to(sae.W_enc.device)
    resample_acts = resample_acts / torch.norm(resample_acts, dim=-1, keepdim=True)

    sae.W_enc.data[:, dead_idxs] = resample_acts.T
    sae.W_dec.data[dead_idxs] = resample_acts
    sae.b_enc.data[dead_idxs] = 0.0

    # set norm of encoder to avg norm of non-dead features * 0.2
    avg_encoder_norm = torch.norm(sae.W_enc.data[:, ~dead_idxs], dim=0).mean()
    sae.W_enc.data[:, dead_idxs] *= avg_encoder_norm * 0.2


def reset_optimizer(sae: SparseAutoencoder, optimizer, dead_idxs):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p is sae.W_enc:
                state = optimizer.state[p]
                state['exp_avg'][:, dead_idxs] = 0.0
                state['exp_avg_sq'][:, dead_idxs] = 0.0
                state['step'] = torch.tensor(0, dtype=torch.float32)
            elif p is sae.W_dec or p is sae.b_enc:
                state = optimizer.state[p]
                state['exp_avg'][dead_idxs] = 0.0
                state['exp_avg_sq'][dead_idxs] = 0.0
                state['step'] = torch.tensor(0, dtype=torch.float32)


def tune_resample(sae: SparseAutoencoder, buffer: ActivationBuffer, dead_idxs):
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-4)
    n_tune_steps = 200

    for _ in range(n_tune_steps):
        optimizer.zero_grad()
        acts = buffer.get_activations()  # [4096, 512]
        acts = acts.to(sae.W_enc.device)
        sae_out, feature_acts, loss, mse_loss, l1_loss = sae(acts)
        loss.backward()

        # only tune the resampled features
        for group in optimizer.param_groups:
            for p in group['params']:
                if p is sae.W_enc:
                    p.grad[:, ~dead_idxs] = 0.0
                elif p is sae.W_dec or p is sae.b_enc:
                    p.grad[~dead_idxs] = 0.0

        optimizer.step()
    optimizer.zero_grad()


if __name__ == "__main__":
    main()
