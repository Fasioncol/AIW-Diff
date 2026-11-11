
```
data/train/ct    data/train/cta
data/val/ct      data/val/cta
data/test/ct     data/test/cta
```

```
conda env create -f environment.yml
conda activate DDIM
pip install scikit-image
```

```
python main.py --config ct_cta.yml --exp exp --doc ct_cta_512 --ni
```

```
python validate.py --config ct_cta.yml --ckpt exp/logs/ct_cta_512/ckpt.pth --data_dir ./data/val
```

```
python test.py --config ct_cta.yml --ckpt exp/logs/ct_cta_512/ckpt.pth --data_dir ./data/test
```


