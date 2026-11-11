import os
import argparse
import yaml
import torch
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.diffusion import Model
from datasets.ct_cta_dataset import CT_CTADataset
from datasets import data_transform, inverse_data_transform
from functions.denoising import generalized_steps
from main import dict2namespace

def validate():
    parser = argparse.ArgumentParser(description="Model Validation on CT-CTA dataset")
    parser.add_argument("--config", type=str, default="ct_cta.yml", help="Path to the config file.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the model checkpoint file (e.g., ckpt_5000.pth).")
    parser.add_argument("--data_dir", type=str, default="./data/val", help="Path to the validation data directory.")
    parser.add_argument("--output_dir", type=str, default="./validation_results",
                        help="Directory to save validation results.")
    parser.add_argument("--timesteps", type=int, default=100, help="Number of steps for sampling.")
    parser.add_argument("--eta", type=float, default=0.0, help="Eta for DDIM sampling (0.0 for deterministic).")
    args = parser.parse_args()

    print("--- Starting Model Validation ---")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Validation Data: {args.data_dir}")

    with open(os.path.join("configs", args.config), "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config.device = device
    print(f"Using device: {device}")

    val_ct_dir = os.path.join(args.data_dir, 'ct')
    val_cta_dir = os.path.join(args.data_dir, 'cta')

    val_dataset = CT_CTADataset(
        ct_dir=val_ct_dir,
        cta_dir=val_cta_dir,
        image_size=config.data.image_size
    )

    model = Model(config)
    states = torch.load(args.ckpt, map_location=device)

    if config.model.ema:
        from models.ema import EMAHelper
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(model)
        print("Loaded EMA model weights for validation.")
    else:
        model.load_state_dict(states[0])
        print("Loaded standard model weights for validation.")

    model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0

    from runners.diffusion import get_beta_schedule
    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)

    print(f"Starting validation on {len(val_dataset)} images...")
    for idx in tqdm(range(len(val_dataset))):

        condition_img, ground_truth_img, complexity_map = val_dataset[idx]

        condition_img = condition_img.unsqueeze(0).to(device)
        ground_truth_img = ground_truth_img.unsqueeze(0).to(device)
        complexity_map = complexity_map.unsqueeze(0).to(device)

        with torch.no_grad():
            condition_transformed = data_transform(config, condition_img)
            ground_truth_transformed = data_transform(config, ground_truth_img)
            noise = torch.randn_like(ground_truth_transformed)

            skip = config.diffusion.num_diffusion_timesteps // args.timesteps
            seq = range(0, config.diffusion.num_diffusion_timesteps, skip)

            generated_sequence, _ = generalized_steps(noise, seq, model, betas,
                                                      condition_transformed,
                                                      complexity_map,
                                                      eta=args.eta)
            generated_img_transformed = generated_sequence[-1]

        ground_truth_to_eval = inverse_data_transform(config, ground_truth_transformed)
        generated_to_eval = inverse_data_transform(config, generated_img_transformed)
        condition_to_save = inverse_data_transform(config, condition_transformed)

        real_np = ground_truth_to_eval.squeeze().cpu().numpy()
        fake_np = generated_to_eval.squeeze().cpu().numpy()

        total_psnr += psnr(real_np, fake_np, data_range=1.0)
        total_ssim += ssim(real_np, fake_np, data_range=1.0, channel_axis=None)

        os.makedirs(args.output_dir, exist_ok=True)
        comparison_grid = torch.cat([
            condition_to_save.cpu(),
            generated_to_eval.cpu(),
            ground_truth_to_eval.cpu()
        ], dim=-1)

        output_filename = f"val_comparison_{idx:04d}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        save_image(comparison_grid, output_path)

    avg_psnr = total_psnr / len(val_dataset)
    avg_ssim = total_ssim / len(val_dataset)

    print("\n--- Validation Finished ---")
    print(f"Checkpoint evaluated: {os.path.basename(args.ckpt)}")
    print(f"Total images evaluated: {len(val_dataset)}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Visual comparison images saved in: {args.output_dir}")

if __name__ == '__main__':
    validate()
