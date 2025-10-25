### WORK IN PROGRESS

# Dynamic Flow Networks (DFN)

**A New Continuous Flow-Inspired Architecture for Next-Gen AI**

---

> “From discrete attention to a seamless flow of information. DFN is not just a model, but a paradigm shift for deep learning.”

---

## 🚀 What is DFN?

Dynamic Flow Networks (DFN) reimagines deep learning by replacing the classical building blocks of transformers—discrete tokens, dot-product attention, and fixed positional encodings—with smooth, *continuous fields* and adaptive semantic flows. Instead of making information “hop” between rigid blocks, DFN lets meaning *flow*, interact, and emerge, much like a swarm of particles in a dynamic field.

---

## 🧑‍🔬 Key Innovations

- **Continuous Entities**: Tokens? Forget about them! DFN samples *entities* from continuous information fields.
- **Dynamic Flow Attention**: Entities move and interact in flexible, learned directions—not just by multiplying queries and keys.
- **FlowNorm & Persistent Field Memory**: Smart normalization and adaptive, long-range context storage.
- Sub-quadratic complexity, true multimodal versatility, and *physics vibes*.

---

## 🔬 High-level Architecture

### 1. Information as a Field
Each input maps to a field \( F(x) \) from which $N$ *entities* are adaptively sampled:

$$
e_i = [p_i, s_i, w_i], \quad i \in [1, N]
$$

where:
- $p_i \in \mathbb{R}^k$: position (space/time)
- $s_i \in \mathbb{R}^d$: latent state
- $w_i$: influence weight (importance)

More entities are extracted where $F(x)$ is information-rich.

### 2. Dynamic Flow Attention (DFA)
Entities produce flow vectors and "drift" into new semantic regions:

$$
f_i = W_f s_i
$$
$$
p_i' = p_i + \alpha f_i
$$

Interactions are *local* (sub-quadratic!). Each entity communicates with its $k$ nearest neighbors using a Gaussian affinity function:

$$
y_i = \sum_{j \in \mathcal{N}(i)} g(\|p_i' - p_j'\|) \cdot W_v s_j
$$

where $g(\cdot)$ is a Gaussian kernel.

### 3. Flow Normalization
To stabilize things (no wild flows!), we use:

$$
\text{FlowNorm}(s_i) = \frac{s_i}{\sqrt{\mathbb{E}[\|f_i\|^2] + \epsilon}}
$$

---

## ✨ Why DFN?

- **No token boundaries or position encodings**
- **Local and global context merged in continuous space**
- **Efficient O(n log n) scaling**
- **Unified for text, images, audio, and beyond**
- Fits your brain like physics, but it’s all deep learning!

---

## ⚡ Training and Usage

- **Modes:** Autoregressive, contrastive, or self-supervised learning
- **Losses:** Language modeling ($\text{softmax}(W_o \bar{s})$), field reconstruction (for images/audio)
- **Optimizers:** AdamW + warmup/cosine decay recommended

---

## 🧪 Example: PyTorch (Pseudocode)

```python
import torch, torch.nn as nn
class DFNLayer(nn.Module):
    def __init__(self, dim, k=16, alpha=0.1):
        super().__init__()
        self.flow = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(2*dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.alpha = alpha
        self.k = k
    def forward(self, states, positions):
        f = self.flow(states)
        p_new = positions + self.alpha * f
        neighbors = knn_search(p_new, self.k)
        affinities = gaussian_affinity(p_new, neighbors)
        context = aggregate(affinities, self.value(states))
        out = self.mlp(torch.cat([states, context], dim=-1))
        return FlowNorm(out), p_new
```

---

## 🎯 Experiments & Future Directions

- **Language modeling:** Outscale transformers for very long sequences (>32k tokens)
- **Vision, audio, and multimodal:** Unified deep learning
- **Ablation:** What if you remove flows, persistent memory, or normalization?

If you want to break the boundaries of symbolic computation and let deep learning “flow” naturally, DFN is your playground.

---

## 🚀 Performance Insights & VRAM Usage

Below is a sample output from `demo.py` showcasing DFN's performance and VRAM usage during scalability tests. Note the significant increase in processing time for very long sequences, which indicates that the dedicated GPU VRAM capacity might have been exceeded, forcing the system to utilize slower shared system RAM.

```

⚡ PERFORMANCE BENCHMARK
==============================
📏 Scalability test:
   Length   32: 0.034s, Current VRAM: 59.0MB, Reserved: 96.0MB, ΔAlloc: 0.1MB, ΔReserved: 22.0MB
   Length   64: 0.033s, Current VRAM: 59.0MB, Reserved: 96.0MB, ΔAlloc: 0.0MB, ΔReserved: 22.0MB
   Length  128: 0.027s, Current VRAM: 59.2MB, Reserved: 96.0MB, ΔAlloc: 0.0MB, ΔReserved: 22.0MB
   Length  256: 0.028s, Current VRAM: 59.4MB, Reserved: 98.0MB, ΔAlloc: 0.0MB, ΔReserved: 24.0MB
   Length  512: 0.025s, Current VRAM: 59.9MB, Reserved: 98.0MB, ΔAlloc: 0.0MB, ΔReserved: 22.0MB
   Length 204800: 0.296s, Current VRAM: 458.9MB, Reserved: 2886.0MB, ΔAlloc: 0.0MB, ΔReserved: 2410.0MB
   Length 256000: 0.429s, Current VRAM: 558.9MB, Reserved: 3986.0MB, ΔAlloc: 0.0MB, ΔReserved: 3012.0MB
   Length 307200: 1.509s, Current VRAM: 658.9MB, Reserved: 4790.0MB, ΔAlloc: 0.0MB, ΔReserved: 3616.0MB
   Length 358400: 5.220s, Current VRAM: 758.9MB, Reserved: 5592.0MB, ΔAlloc: 0.0MB, ΔReserved: 4218.0MB
   Length 409600: 10.885s, Current VRAM: 858.9MB, Reserved: 6394.0MB, ΔAlloc: 0.0MB, ΔReserved: 4820.0MB
   Length 460800: 13.353s, Current VRAM: 958.9MB, Reserved: 7196.0MB, ΔAlloc: 0.0MB, ΔReserved: 5422.0MB
   Length 512000: 13.663s, Current VRAM: 1058.9MB, Reserved: 7996.0MB, ΔAlloc: 0.0MB, ΔReserved: 6022.0MB
   Length 563200: 15.506s, Current VRAM: 1158.9MB, Reserved: 8800.0MB, ΔAlloc: 0.0MB, ΔReserved: 6626.0MB
   Length 614400: 17.125s, Current VRAM: 1258.9MB, Reserved: 9602.0MB, ΔAlloc: 0.0MB, ΔReserved: 7228.0MB
   Length 665600: 18.507s, Current VRAM: 1358.9MB, Reserved: 10404.0MB, ΔAlloc: 0.0MB, ΔReserved: 7830.0MB
   Length 716800: 20.203s, Current VRAM: 1458.9MB, Reserved: 8506.0MB, ΔAlloc: 0.0MB, ΔReserved: 5732.0MB
   Length 768000: 21.351s, Current VRAM: 1558.9MB, Reserved: 9106.0MB, ΔAlloc: 0.0MB, ΔReserved: 6132.0MB
```

*Peak VRAM utilization observed during this test was approximately **1558.9MB**.*

---

## 👋 Get Involved

- Star ⭐ the repo if you like the idea!
- Open an issue or PR if you want to contribute - especially new field samplers, training tricks, or hardware-focused variants.

**Contact:** [joaquinsturtz26@gmail.com] | [[instagram](https://www.instagram.com/joaco_sturtz/)]

---

> **DFN**: “Let your data flow.”
