import os
import logging
import time

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from datasets.ct_cta_dataset import CT_CTADataset
from functions.denoising import generalized_steps
from functions.sampler.timestep_sampler import AsymmetricTimeStepSampler
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.device = device

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)

        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        pi = torch.sqrt(
            (self.alphas_cumprod_prev / self.alphas_cumprod) * (1 - self.alphas_cumprod) / (
                        1 - self.alphas_cumprod_prev)
        ) * (1 - self.alphas_cumprod_prev) / torch.sqrt(1 - self.alphas_cumprod)
        pi_t_1 = torch.sqrt(1 - self.alphas_cumprod_prev)

        pi[0] = pi_t_1[1] - pi_t_1[0]
        pi[1:-1] = pi_t_1[2:] - pi_t_1[1:-1]
        pi[-1] = pi[-2]

        self.pi = pi.to(self.device)
        self.pi_avg = self.pi.mean()

        self.w_p = (self.pi / self.pi_avg).to(self.device)
        logging.info("Initialized SpeeD process-increment weights.")

    def train(self):
        config = self.config
        tb_logger = self.config.tb_logger
        dataset = get_dataset(config)
        train_loader = data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                                       num_workers=config.data.num_workers)

        model = Model(config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        sampler = None
        if hasattr(config.training, 'speed') and config.training.speed.use:
            sampler = AsymmetricTimeStepSampler(num_timesteps=self.num_timesteps, lam=config.training.speed.lam,
                                                k=config.training.speed.k)
        logging.info("Using SpeeD asymmetric timestep sampler.")

        optimizer = get_optimizer(self.config, model.parameters())

        ema_helper = EMAHelper(mu=self.config.model.ema_rate) if self.config.model.ema else None
        if ema_helper:
            ema_helper.register(model)

        val_root = config.training.val_data_dir
        val_dataset = CT_CTADataset(
            ct_dir=os.path.join(val_root, "ct"),
            cta_dir=os.path.join(val_root, "cta"),
            image_size=config.data.image_size,
        )
        val_num = min(int(config.training.val_num_samples), len(val_dataset))
        logging.info(
            f"Every {config.training.snapshot_freq} steps, save weights and validate "
            f"the first {val_num}/{len(val_dataset)} pairs from {val_root}."
        )

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if ema_helper:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0

            for i, (condition, x, complexity_map) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)
                condition = condition.to(self.device)
                complexity_map = complexity_map.to(self.device)

                x = data_transform(self.config, x)
                condition = data_transform(self.config, condition)

                e = torch.randn_like(x)
                b = self.betas

                w_t = None
                w_p_blended = None
                if sampler:
                    t, w_t = sampler.sample(n, self.device)
                    w_t = w_t.to(self.device)

                    wp_strength = config.training.speed.wp_strength if hasattr(config.training.speed,
                                                                               'wp_strength') else 1.0
                    raw_wp = self.w_p[t]
                    w_p_blended = (1.0 - wp_strength) + wp_strength * raw_wp
                else:
                    t = torch.randint(low=0, high=self.num_timesteps, size=(n,)).to(self.device)

                use_eabc = config.model.eabc.use if hasattr(config.model, 'eabc') else False
                eabc_weight = config.model.eabc.loss_weight if use_eabc else 0.0
                l1_weight = config.model.eabc.l1_loss_weight if use_eabc else 0.0

                loss = loss_registry[config.model.type](
                    model, x, t, e, b, condition,
                    complexity_map,
                    ib_loss_weight=config.model.ib_mlp.loss_weight,
                    eabc_loss_weight=eabc_weight,
                    l1_loss_weight=l1_weight,
                    w_t=w_t,
                    w_p=w_p_blended
                )

                tb_logger.add_scalar("loss", loss, global_step=step)
                logging.info(f"step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}")

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
                except Exception:
                    pass
                optimizer.step()

                if ema_helper:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0:
                    states = [model.state_dict(), optimizer.state_dict(), epoch, step]
                    if ema_helper:
                        states.append(ema_helper.state_dict())
                    torch.save(states, os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)))
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    logging.info(f"step: {step}, checkpoint saved")
                    self._validate_and_resume(
                        model, ema_helper, val_dataset, val_num, step, tb_logger
                    )

                data_start = time.time()

    def _validate_and_resume(self, model, ema_helper, val_dataset, val_num, step, tb_logger):
        config = self.config
        timesteps = int(config.training.val_timesteps)
        eta = float(config.training.val_eta)
        skip = config.diffusion.num_diffusion_timesteps // timesteps
        seq = list(range(0, config.diffusion.num_diffusion_timesteps, skip))

        raw_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        total_psnr = 0.0
        total_ssim = 0.0
        try:
            if ema_helper is not None:
                ema_helper.ema(model)
                logging.info(f"step: {step}, validating with EMA weights")
            else:
                logging.info(f"step: {step}, validating with current weights")
            model.eval()
            logging.info(
                f"step: {step}, start validation on {val_num} pairs, timesteps={timesteps}, eta={eta}"
            )

            for idx in tqdm.tqdm(range(val_num), desc=f"val@{step}"):
                condition_img, ground_truth_img, complexity_map = val_dataset[idx]
                condition_img = condition_img.unsqueeze(0).to(self.device)
                ground_truth_img = ground_truth_img.unsqueeze(0).to(self.device)
                complexity_map = complexity_map.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    condition_transformed = data_transform(config, condition_img)
                    ground_truth_transformed = data_transform(config, ground_truth_img)
                    noise = torch.randn_like(ground_truth_transformed)
                    generated_sequence, _ = generalized_steps(
                        noise,
                        seq,
                        model,
                        self.betas,
                        condition_transformed,
                        complexity_map,
                        eta=eta,
                    )
                    generated_img_transformed = generated_sequence[-1]

                ground_truth_to_eval = inverse_data_transform(config, ground_truth_transformed)
                generated_to_eval = inverse_data_transform(config, generated_img_transformed)
                real_np = ground_truth_to_eval.squeeze().detach().cpu().numpy()
                fake_np = generated_to_eval.squeeze().detach().cpu().numpy()
                total_psnr += psnr(real_np, fake_np, data_range=1.0)
                total_ssim += ssim(real_np, fake_np, data_range=1.0, channel_axis=None)

            avg_psnr = total_psnr / val_num
            avg_ssim = total_ssim / val_num
            logging.info(
                f"step: {step}, validation images: {val_num}, "
                f"Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}"
            )
            tb_logger.add_scalar("val_psnr", avg_psnr, global_step=step)
            tb_logger.add_scalar("val_ssim", avg_ssim, global_step=step)
        finally:
            model.load_state_dict(raw_state)
            model.train()
            del raw_state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"step: {step}, restored training weights, continue training")
