# CT 条件生成 CTA

用 512×512 的平扫 CT 生成对应 CTA。DICOM 按文件名排序后配对：

```
data/train/ct    data/train/cta
data/val/ct      data/val/cta
data/test/ct      data/test/cta
```

```
conda env create -f environment.yml
conda activate DDIM
pip install scikit-image
```

训练：

```
python main.py --config ct_cta.yml --exp exp --doc ct_cta_512 --ni
```

验证：

```
python validate.py --config ct_cta.yml --ckpt exp/logs/ct_cta_512/ckpt_5000.pth --data_dir ./data/val
```

权重保存在 `exp/logs/ct_cta_512/`。验证结果保存在 `validation_results/`，并输出平均 PSNR 和 SSIM。

