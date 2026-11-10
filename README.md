# signs

Edge-detection front ends (Sobel, Canny, learned) feeding dense, quantized (QAT)
and sparse CNNs on GTSRB (43 traffic-sign classes).

## Colab

```python
!pip install -q git+https://github.com/<you>/signs.git

import signs

# quick smoke test on random data (seconds)
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
