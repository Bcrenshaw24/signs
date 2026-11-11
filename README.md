# signs

Edge-detection front ends (Sobel, Canny, learned) feeding dense, quantized (QAT)
and sparse CNNs on GTSRB (43 traffic-sign classes).

## Colab

```python
!pip install -q git+https://github.com/<you>/signs.git

import signs

# test before training & eval
signs.run(dataset="fake", epochs=1, subset=512)

# real run (downloads GTSRB ~350 MB on first call; use a GPU runtime)
results = signs.run(epochs=15)
```

`run()` takes the same options as the CLI, with dashes as underscores, e.g.

```python
signs.run(filters=["learned"], models=["quant", "sparse"],
          w_bits=[8, 4, 2, 1], sparsity=[0.8, 0.9, 0.95])
```

Results go to `results/` (`results.csv`, `results.json`, `filters.png`, `ckpt/`).
To view them in Colab:

```python
import pandas as pd; from IPython.display import Image
display(pd.DataFrame(results)); display(Image("results/filters.png"))
```

## Local

```bash
pip install -e .
python -m signs.data          # download + cache the dataset
python -m signs.experiment    # full comparison
python -m signs.experiment --dataset fake --epochs 1 --subset 512
```

## Results (Benchmarked)

### Sobel

| Filter | Model | Test Accuracy | Speed vs dense (ms/img) | Size vs dense (KB) |
|---|---|---|---|---|
| sobel | dense-fp32 | 99.11% | 100% (0.0335) | 100.0% (3217.1) |
| sobel | quant-w4a4 | 98.81% | 66% (0.0510) | 13.0% (416.7) |
| sobel | sparse-90 | 98.95% | 95% (0.0351) | 15.4% (495.8) |
| sobel | sparse-90-csr | 98.95% | 25% (0.1326) | 15.4% (495.8) |

### Canny

| Filter | Model | Test Accuracy | Speed vs dense (ms/img) | Size vs dense (KB) |
|---|---|---|---|---|
| canny | dense-fp32 | 87.36% | 100% (0.0401) | 100.0% (3214.8) |
| canny | quant-w4a4 | 87.21% | 71% (0.0561) | 12.9% (416.2) |
| canny | sparse-90 | 86.55% | 100% (0.0401) | 15.3% (492.4) |
| canny | sparse-90-csr | 86.55% | 32% (0.1265) | 15.3% (492.4) |

### Learned

| Filter | Model | Test Accuracy | Speed vs dense (ms/img) | Size vs dense (KB) |
|---|---|---|---|---|
| learned | dense-fp32 | 99.07% | 100% (0.0371) | 100.0% (3223.6) |
| learned | quant-w4a4 | 98.65% | 73% (0.0511) | 13.0% (419.0) |
| learned | sparse-90 | 98.74% | 102% (0.0363) | 15.7% (505.1) |
| learned | sparse-90-csr | 98.74% | 26% (0.1423) | 15.7% (505.1) |

Speed and size are relative to that filter's dense-fp32 model (100%). Speed above
100% is faster than dense, below is slower; size below 100% is smaller. Speed is
GPU (CUDA) time per image at batch size 256, including the edge filter. The
quantized model uses fake quantization in fp32, so its speed does not reflect
real low-bit kernels.
