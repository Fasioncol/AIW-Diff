import torch
from .ct_cta_dataset import CT_CTADataset

def get_dataset(config):
    if config.data.dataset != "CT_CTA":
        raise ValueError(f"本精简项目不支持数据集 {config.data.dataset}")

    return CT_CTADataset(
        ct_dir="./data/train/ct",
        cta_dir="./data/train/cta",
        image_size=config.data.image_size,
    )

def data_transform(config, X):
    return X / 127.5 - 1.0

def inverse_data_transform(config, X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)
