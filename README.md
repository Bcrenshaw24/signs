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
| sobel | dense-fp32 | 98.96% | 100% (1.23) | 100.0% (3217.1) |
| sobel | quant-w4a4 | 98.84% | 44% (2.78) | 13.0% (416.7) |
| sobel | sparse-90 | 98.64% | 96% (1.27) | 15.4% (495.8) |
| sobel | sparse-90-csr | 98.64% | 14% (8.79) | 15.4% (495.8) |

### Canny

| Filter | Model | Test Accuracy | Speed vs dense (ms/img) | Size vs dense (KB) |
|---|---|---|---|---|
| canny | dense-fp32 | 87.11% | 100% (1.76) | 100.0% (3214.8) |
| canny | quant-w4a4 | 86.84% | 68% (2.58) | 12.9% (416.2) |
| canny | sparse-90 | 86.82% | 78% (2.25) | 15.3% (492.4) |
| canny | sparse-90-csr | 86.82% | 21% (8.41) | 15.3% (492.4) |

### Learned

| Filter | Model | Test Accuracy | Speed vs dense (ms/img) | Size vs dense (KB) |
|---|---|---|---|---|
| learned | dense-fp32 | 98.76% | 100% (1.89) | 100.0% (3223.6) |
| learned | quant-w4a4 | 98.66% | 95% (1.98) | 13.0% (419.0) |
| learned | sparse-90 | 98.80% | 143% (1.32) | 15.7% (505.1) |
| learned | sparse-90-csr | 98.80% | 23% (8.38) | 15.7% (505.1) |

Speed and size are relative to that filter's dense-fp32 model (100%). Speed above
100% is faster than dense, below is slower; size below 100% is smaller. Speed is
CPU time per image, including the edge filter. The quantized model uses fake
quantization in fp32, so its speed does not reflect real low-bit kernels.
