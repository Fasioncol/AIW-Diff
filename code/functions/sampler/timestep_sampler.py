import numpy as np
import torch

class AsymmetricTimeStepSampler:
    def __init__(self, num_timesteps, lam, k, device='cpu'):
        self.num_timesteps = num_timesteps
        self.lam = lam
        self.k = k
        self.device = device
        self.uniform_sampler = UniformTimeStepSampler(num_timesteps, device)

        self.accel_end = int(num_timesteps * 0.25)
        self.decel_end = int(num_timesteps * 0.75)

    def sample(self, batch_size, device):
        if self.num_timesteps <= 10:
            return self.uniform_sampler.sample(batch_size, device), torch.ones(batch_size, device=device)

        t = np.random.choice(self.num_timesteps, batch_size)
        p = self._get_p(t)
        weights = 1 / (p * self.num_timesteps)
        return torch.from_numpy(t).long().to(device), torch.from_numpy(weights).float().to(device)

    def _get_p(self, t):
        if isinstance(t, torch.Tensor):
            t = t.cpu().numpy()
        accel_mask = (t >= 0) & (t < self.accel_end)
        decel_mask = (t >= self.accel_end) & (t < self.decel_end)
        conver_mask = t >= self.decel_end

        p = np.zeros_like(t, dtype=np.float32)
        p[accel_mask] = self._accel_p(t[accel_mask])
        p[decel_mask] = self._decel_p(t[decel_mask])
        p[conver_mask] = self._conver_p(t[conver_mask])
        return p

    def _accel_p(self, t):
        return (self.lam + 1) / (self.lam * self.accel_end + self.num_timesteps - self.accel_end)

    def _decel_p(self, t):
        return 1 / (self.lam * self.accel_end + self.num_timesteps - self.accel_end) * \
               (1 + (self.lam - 1) / (self.decel_end - self.accel_end) * (self.decel_end - t))

    def _conver_p(self, t):
        return 1 / (self.lam * self.accel_end + self.num_timesteps - self.accel_end) * \
               (1 / self.k + (1 - 1 / self.k) / (self.num_timesteps - self.decel_end) * (self.num_timesteps - t))

class UniformTimeStepSampler:
    def __init__(self, num_timesteps, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device

    def sample(self, batch_size, device):
        t = np.random.choice(self.num_timesteps, batch_size)
        return torch.from_numpy(t).long().to(device)
