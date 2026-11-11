import torch

def eabc_loss(pred_clean, ground_truth, device):
    laplacian_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                                    dtype=torch.float32, device=device).reshape(1, 1, 3, 3)

    pred_edges = torch.nn.functional.conv2d(pred_clean, laplacian_kernel, padding=1)
    gt_edges = torch.nn.functional.conv2d(ground_truth, laplacian_kernel, padding=1)

    theta = 0.1
    B = (pred_edges >= theta).float()
    P = (gt_edges >= theta).float()

    epsilon = 1e-8
    loss = -torch.sum(
        P * torch.log(B + epsilon) + (1 - P) * torch.log(1 - B + epsilon)
    )

    return loss / pred_clean.size(0)

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          condition: torch.Tensor,
                          complexity_map: torch.Tensor,
                          ib_loss_weight: float = 1e-5,
                          eabc_loss_weight: float = 0.0,
                          l1_loss_weight: float = 0.0,
                          w_t: torch.Tensor = None,
                          w_p: torch.Tensor = None):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    model_input = torch.cat([x, condition], dim=1)

    predicted_noise, mu, log_var, predicted_clean_image = model(model_input, t.float(), complexity_map)

    loss_denoise_per_sample = (e - predicted_noise).square().sum(dim=(1, 2, 3))
    loss_kl_per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    combined_loss_per_sample = loss_denoise_per_sample + ib_loss_weight * loss_kl_per_sample

    loss_l1 = torch.nn.functional.l1_loss(predicted_clean_image, x0)
    loss_eabc = eabc_loss(predicted_clean_image, x0, x0.device)

    if w_t is not None and w_p is not None:
        w_t = w_t.view(-1)
        w_p = w_p.view(-1)
        weighted_denoise_kl_loss = (combined_loss_per_sample * w_t * w_p).mean()

        total_loss = weighted_denoise_kl_loss + \
                     l1_loss_weight * loss_l1 + \
                     eabc_loss_weight * loss_eabc
    else:
        total_loss = combined_loss_per_sample.mean() + \
                     l1_loss_weight * loss_l1 + \
                     eabc_loss_weight * loss_eabc

    return total_loss

loss_registry = {
    'hybrid': noise_estimation_loss,
}
