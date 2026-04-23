import torch
import torch.nn.functional as F


def dreambooth_loss(
    unet,
    scheduler,
    subject_latents,
    subject_encoder_hidden_states,
    prior_latents,
    prior_encoder_hidden_states,
    device,
    lam: float = 1.0,
):
    """
    L_DB = E[||eps_theta(z_t, t, c) - eps||^2]
         + lambda * E[||eps_theta(z_pr_t, t, c_pr) - eps||^2]
    """
    bsz = subject_latents.shape[0]
    bsz_pr = prior_latents.shape[0]

    t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
    t_pr = torch.randint(0, scheduler.config.num_train_timesteps, (bsz_pr,), device=device).long()

    noise = torch.randn_like(subject_latents)
    noise_pr = torch.randn_like(prior_latents)

    z_t = scheduler.add_noise(subject_latents, noise, t)
    z_pr_t = scheduler.add_noise(prior_latents, noise_pr, t_pr)

    noise_pred = unet(z_t, t, encoder_hidden_states=subject_encoder_hidden_states).sample
    noise_pred_pr = unet(z_pr_t, t_pr, encoder_hidden_states=prior_encoder_hidden_states).sample

    loss_subject = F.mse_loss(noise_pred, noise, reduction="mean")
    loss_prior = F.mse_loss(noise_pred_pr, noise_pr, reduction="mean")

    return loss_subject + lam * loss_prior
