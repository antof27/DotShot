---
name: pytorch-cuda-engineering
description: >-
  Optimize PyTorch RL and MARL training on Windows with NVIDIA CUDA (RTX 3060
  12GB), including CPU thread isolation for SubprocVecEnv, orthogonal weight
  initialization, float32 numerical precision, and deterministic seeding.
---

# PyTorch & CUDA Engineering for RL/MARL on Windows

Use this skill when building or tuning PyTorch networks, rollout buffers, and training loops under `benchmarks/marl/` and `benchmarks/runners/`.

---

## 1. Windows Multiprocessing & Threading Isolation

- **Prevent CPU Core Thrashing**: Inside vectorized CPU/GPU training scripts, call `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)` (when safe) before spawning `SubprocVecEnv` workers. Leaving PyTorch at default thread counts inside 8+ parallel workers causes severe OpenMP contention on Windows.
- **Entry-Point Guard**: Always protect `SubprocVecEnv` creation behind `if __name__ == "__main__":` so Windows `spawn` workers do not recursively re-execute training.

---

## 2. RTX 3060 (12 GB VRAM) Allocation & Precision

- **Keep Policy/Value Math in `float32`**: Do not force `torch.float16` AMP on small RL MLPs (`128` to `256` width) or GAE/ratio computations. Half-precision rounding destabilizes PPO clipped ratios $\exp(\log \pi_{\text{new}} - \log \pi_{\text{old}})$ and QMIX target TD errors.
- **Batch on GPU, Step on CPU**: Keep environment stepping in vectorized NumPy workers on CPU and transfer batched tensors (`[T, B, ...]` or `[B, ...]`) to `cuda` once per rollout or minibatch update.
- **Memory Discipline**: Call `torch.cuda.empty_cache()` only between independent multi-seed runs if needed, never inside inner training loops. Always wrap inference/evaluation and rollout collection in `with torch.no_grad():`.

---

## 3. RL Network Stability Defaults

- **Orthogonal Initialization**: Apply `nn.init.orthogonal_(layer.weight, gain=...)` with `gain=np.sqrt(2)` for hidden ReLU/Tanh layers, `gain=0.01` for actor output heads (preventing premature high-confidence action collapse), and `gain=1.0` for critic value heads.
- **Gradient Clipping**: Always clip global gradient norm (`nn.utils.clip_grad_norm_(params, max_grad_norm=0.5)` for PPO/MAPPO/HAPPO/MAT, `10.0` for QMIX) and monitor post-clip norm in TensorBoard.
- **Safe Log-Probs & Entropy**: Use `torch.distributions.Categorical(logits=...)` (never raw `probs=...`) so PyTorch applies log-sum-exp numerical stabilization automatically.

---

## 4. Deterministic CUDA Seeding

For reproducible benchmark runs:

```python
import os, random, numpy as np, torch

def seed_everything(seed: int, deterministic_cuda: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```
