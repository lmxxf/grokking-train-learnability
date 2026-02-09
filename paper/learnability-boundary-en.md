# Learnability Boundary: How Complex Can Neural Networks Learn Pseudo-Random Sequences?

**Authors**: Jin Yanyan (lmxxf@hotmail.com), Zhao Lei (zhaosanshi@gmail.com)

**Abstract**: Classical information theory assumes observers have unlimited computational power, making pseudo-random numbers "low entropy" (seeds are only a few hundred bits). However, for computationally bounded neural networks, the same data may be "completely unlearnable noise." This paper experimentally explores the boundary of neural network learnability: from simple periodic patterns to cryptographically secure random numbers, we identify where Epiplexity (extractable structure) transitions from non-zero to zero. We train identical Transformer architectures on 6 sequences of varying complexity and find: (1) pseudo-random sequences with state space $\leq 256$ (simple LCG, 5-bit LFSR) can be perfectly learned; (2) pseudo-random with state space 2³¹ (glibc LCG) is completely unlearnable, with test accuracy equal to random guessing; (3) 31-bit LFSR exhibits **partial Grokking**—the model learns 7 out of 8 bits of the sequence's pattern (50% accuracy on 256-class classification) but cannot learn the last 1 bit. Further model scale ablation experiments reveal a counter-intuitive phenomenon: scaling up the model (from 0.3M to 33M parameters) not only fails to break the learnability boundary, but causes LFSR-31's partial Grokking to collapse entirely—**the larger model loses the 7 bits the small model learned**. Reducing weight decay by 50x (from 0.5 to 0.01) cannot rescue this collapse, demonstrating the problem lies in the instability of the optimization landscape in larger models, not excessive regularization. A context window expansion experiment finally reveals the true cause of the 50% ceiling: increasing context_len from 16 to 32/64, LFSR-31's test accuracy jumps from 50% → 99.8% → 100%—**the bottleneck was not model capacity, but input information**. The additional pattern redundancy from longer windows eliminates residual errors. A data scaling experiment further validates the root cause of LCG's unlearnability: expanding lcg_glibc's data by 20x (10k→200k) still yields zero generalization (test 0.3%), while training accuracy drops from 100% to 22%—**when no learnable manifold exists, more data is meaningless**. These findings validate the core prediction of the Epiplexity paper: **information is observer-dependent; the same data presents different learnability to observers with different computational power**; and further reveal that the learnability boundary is a joint property of task complexity, model capacity, and input information.

---

## 1. Introduction: The 78-Year Blind Spot in Information Theory

In 1948, when Claude Shannon founded information theory, he assumed observers have **unlimited computational power**. Under this assumption, the output of a cryptographic random number generator has "low information"—because the program generating it is only a few lines, and the seed is only 256 bits.

But this assumption never holds in reality.

For a computationally bounded neural network, that same 1GB of pseudo-random data may be "completely incomprehensible pure noise"—it cannot find any pattern no matter how much computation it expends.

In January 2026, a paper from CMU and NYU, "From Entropy to Epiplexity," formally challenged this 78-year-old assumption and proposed a new framework:

$$
\text{Data} = \text{Epiplexity}(S_T) + \text{Time-bounded Entropy}(H_T)
$$

Where:
- **Epiplexity ($S_T$)**: The structure/patterns you can extract from data given computational budget $T$
- **Time-bounded Entropy ($H_T$)**: The part you simply cannot learn

**Core insight**: For the same data, observers with more computational power can see more structure (higher Epiplexity) and less noise (lower Entropy).

This paper aims to **experimentally validate this theory**: using neural networks with identical architecture to learn pseudo-random sequences of different complexity, finding where the learnability boundary lies.

---

## 2. Experimental Design

### 2.1 Core Question

**How "weak" can pseudo-random be for neural networks to still learn it?**

We designed 6 sequence generators ranging from simple to complex:

| Sequence Type | Generation Rule | State Space | Expected |
|--------------|-----------------|-------------|----------|
| periodic | $(i \mod 7) > 3 \to 0/1$ | 7 | Yes |
| lcg_simple | $x_{n+1} = (3x_n + 7) \mod 256$ | 256 | Yes |
| lfsr_5 | $x_n = x_{n-1} \oplus x_{n-5}$ | 32 | Yes |
| lcg_glibc | $x_{n+1} = (1103515245 x_n + 12345) \mod 2^{31}$ | $2^{31}$ | Boundary |
| lfsr_31 | 31-bit LFSR (polynomial $x^{31} + x^3 + 1$) | $2^{31}$ | Boundary |
| csprng | Python `secrets.randbits` (cryptographically secure) | $\infty$ | No |

**Task**: Given the first $k$ tokens of a sequence, predict the $(k+1)$-th (baseline experiments $k=16$, window expansion experiments $k=32, 64$).

### 2.2 Model Configuration

We reuse the architecture from our previous Grokking experiments:

| Parameter | Value |
|-----------|-------|
| Model | 2-layer Transformer |
| Hidden dim | 128 |
| Attention heads | 4 |
| Context length | 16 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1.0 |
| Total steps | 100,000 |
| Training ratio | 30% |
| Total sequence length | 10,000 |

### 2.3 Evaluation Metrics

- **Test accuracy**: Evaluated on 70% of the data
- **Random baseline**: periodic/lfsr_5 = 50% (binary classification), others = 1/256 $\approx$ 0.4%
- **Learnability criterion**: Test accuracy significantly above random baseline

---

## 3. Experimental Results

### 3.1 Main Result: Phase Transition Boundary Between 256 and 2³¹

| Sequence Type | Final Test Accuracy | Random Baseline | Learnable? |
|--------------|---------------------|-----------------|------------|
| periodic | **100.00%** | 50.00% | Yes |
| lcg_simple | **100.00%** | 0.39% | Yes |
| lfsr_5 | **100.00%** | 50.00% | Yes |
| lcg_glibc | 0.41% | 0.39% | No |
| lfsr_31 | **50.16%** | 0.39% | Partial |
| csprng | 0.33% | 0.39% | No |

**Key Findings**:

1. **Sequences with state space $\leq 256$ can be perfectly learned**: periodic, lcg_simple, and lfsr_5 all achieve 100% test accuracy

2. **State space 2³¹ LCG is completely unlearnable**: lcg_glibc test accuracy equals random guessing (0.41% vs 0.39% baseline), while training accuracy is 100%—pure memorization, zero generalization

3. **LFSR-31 exhibits partial Grokking**: 50.16% accuracy is not random; the model learned 7 out of 8 bits, with the last 1 bit remaining unlearnable (see Section 3.2)

4. **CSPRNG is completely unlearnable**: Even training accuracy is only 1%—it cannot even memorize

### 3.2 Unexpected Finding: "Partial Grokking" in LFSR-31

The 50% accuracy of LFSR-31 is quite interesting.

**The random baseline is 0.39% (256-class classification), not 50%**. 50% means the model learned some binary pattern.

Possible explanations:
- The model learned the pattern of the output's **least significant bit** (LSB)
- Or learned some **parity** determination

This suggests Grokking is not all-or-nothing, but rather **"learn as much as you can"**—the model learned 7 out of 8 bits, but couldn't learn the last 1 bit.

**Dimension analysis supports this explanation**:

| Sequence | Average Intrinsic Dimension | Interpretation |
|----------|---------------------------|----------------|
| lcg_glibc | 10-14 dimensions | Medium dimension, but learned nothing |
| **lfsr_31** | **2-4 dimensions** | Extremely low—did capture some simple structure |
| csprng | 10-14 dimensions | Medium dimension, learned nothing |

LFSR-31's representation space collapsed to 2-4 dimensions, indicating the model found an extremely compact low-dimensional structure to encode 7 bits of the pattern, with the last 1 bit beyond its capacity.

### 3.3 Training Dynamics Comparison

| Sequence | Final Train Acc | Final Test Acc | Diagnosis |
|----------|----------------|----------------|-----------|
| periodic | 100% | 100% | Perfect generalization |
| lcg_simple | 100% | 100% | Perfect generalization |
| lfsr_5 | 100% | 100% | Perfect generalization |
| lcg_glibc | **100%** | **0.41%** | Pure memorization, zero generalization |
| lfsr_31 | 99.9% | 50.2% | Partial generalization (7/8 bit) |
| csprng | **1.0%** | 0.33% | Even memorization failed |

**lcg_glibc's "100% train + 0% test" is a textbook "memorization solution"**—the model memorized 3000 training sample input-output mappings, but there's no generalizable structure between these mappings.

---

## 4. Theoretical Explanation

### 4.1 Reinterpreting with the Epiplexity Framework

The core formula from the Epiplexity paper:

$$
P^* = \arg\min_{P \in \mathcal{P}_T} \{|P| + \mathbb{E}[\log 1/P(X)]\}
$$

$$
S_T(X) = |P^*| \quad \text{(Epiplexity)}
$$

In plain language: Within your computational budget, find the optimal balance between "shortest program + most accurate prediction." The length of that program is the Epiplexity.

**Explaining experimental results with this framework**:

| Sequence | State Space | Shortest Program Model Can Find | Epiplexity |
|----------|-------------|--------------------------------|------------|
| periodic | 7 | `(i % 7) > 3` | Very low (few bits) |
| lcg_simple | 256 | `x = (3x + 7) % 256` | Low (tens of bits) |
| lfsr_5 | 32 | `x = x[-1] XOR x[-5]` | Low |
| lcg_glibc | $2^{31}$ | Cannot find | **0** |
| lfsr_31 | $2^{31}$ | 7/8 bit partial pattern | Low (~7 bit) |
| csprng | ∞ | Cannot find | **0** |

**Epiplexity = 0 means**: Within the model's computational budget, there is no extractable structure in the data.

### 4.2 Physical Meaning of the Phase Transition Point

Experiments found the phase transition point between **256 and 2³¹**.

This is not coincidental:
- Model's hidden dimension = 128
- 128-dimensional space can "comfortably" encode $2^7 = 128$ to $2^8 = 256$ states
- But cannot encode $2^{31}$ states—off by 23 orders of magnitude

**In other words**: The model's "memory capacity" is approximately between $2^7$ and $2^8$; beyond this, it can only generalize by "discovering patterns."

- lcg_simple: 256 states, right at the boundary, can learn
- lcg_glibc: $2^{31}$ states, far beyond the boundary, cannot learn

### 4.3 Partial Grokking in LFSR-31: Learn What You Can

The LFSR-31 result suggests a more nuanced picture:

**Grokking is not all-or-nothing, but layered**.

The model may attempt to learn in order of increasing complexity:
1. First learn the simplest pattern (e.g., periodicity of LSB)
2. If successful, try to learn more complex patterns
3. If failed (insufficient computational power), stop at the current level

LFSR-31's 50% accuracy = learned 7 bits (precisely hitting half of 256 classes = 128 classes), stuck at the last 1 bit.

This is consistent with the **two-stage Grokking** we discovered in our modular multiplication experiments:
- Modular multiplication learned the quotient group $\mathbb{Z}_{12}$, not the complete $\mathbb{Z}_{96}$
- LFSR-31 learned 7 bits of the pattern, not the complete 8-bit pattern

**Unified explanation**: Grokking is layered manifold discovery; each layer requires a different computational budget.

### 4.4 Same $2^{31}$ State Space: Why Can LFSR Learn but LCG Cannot?

Both lcg_glibc and lfsr_31 have state spaces of $2^{31}$, yet their learnability is drastically different. This demonstrates that **state space size is not the sole determinant of learnability—the internal geometric structure of the algorithm is equally critical**.

**LFSR (XOR) = Axis-Aligned**. XOR is a linear operation in $GF(2)$ space, where bits are relatively independent and data is arranged on quasi-orthogonal hyperplanes. The Transformer can capture individual bits through low-rank approximation—this is why the model learns 7/8 bits. Although LFSR's state space is large, its entropy is "axis-aligned," permitting partial extraction.

**LCG (Multiplication mod $2^{31}$) = Topological Shredding**. Multiplication makes every output bit depend on all lower-order input bits, crushing what would be a smooth linear manifold into infinitely dense fragments. In the 128-dimensional embedding space, LCG's output points are indistinguishable from white noise—gradients cannot find any locally continuous "handle" to grip. LCG's entropy is "isotropic," causing severe aliasing when projected to lower dimensions.

**The Physical Threshold for Grokking**: Whether the local smoothness of the manifold exceeds the sampling frequency of the model's weights. LFSR's manifold is locally smooth and sampleable; LCG's manifold is shredded by multiplication below the sampling resolution, rendering it invisible.

---

## 5. Connection to Grokking Experiments

This experiment complements our previous Grokking manifold discovery experiments (modular addition/multiplication):

| Dimension | Grokking Experiments | This Experiment |
|-----------|---------------------|-----------------|
| Core question | What happens during Grokking? | What tasks can Grok? |
| Task type | Deterministic mathematical operations | Pseudo-random sequence prediction |
| Focus | Geometric changes in representation space | Boundary of learnability |
| Discovery | Dimension collapse, topological phase transition | State space determines learnability |

**Unified picture from both experiments**:

1. **Grokking = Manifold Discovery**: From high-dimensional memorization curve to low-dimensional structural manifold
2. **Learnability = Whether Manifold Exists**: If there's no low-dimensional manifold behind the data (like CSPRNG), Grokking cannot occur
3. **Partial Grokking = Partial Manifold Discovery**: Both LFSR-31 and modular multiplication only discovered "shallow" manifolds

---

## 6. Discussion

### 6.1 Implications for AI Training

1. **Value of Synthetic Data**: The Epiplexity framework proves that using AI-generated data to train AI is meaningful—computation can "unfold" structure folded in original rules

2. **Data Quality Assessment**: Epiplexity can quantify the "nutritional value" of datasets—high Epiplexity data can teach models more

3. **Model-Data Matching**: Not all data suits all models. Small models cannot learn patterns with large state spaces; forcing it only yields memorization solutions

### 6.2 Model Scale Experiment: Can Larger Models Break the Boundary?

To verify whether the learnability boundary is limited by model capacity, we conducted model scale ablation experiments:

| Model | Parameters | embed_dim | layers | heads |
|-------|------------|-----------|--------|-------|
| Small | 0.3M | 128 | 2 | 4 |
| 4x | 8M | 512 | 4 | 8 |
| 10x | 33M | 1024 | 4 | 16 |

**Results (Phase 3: Scaling up)**:

| Sequence | Small (0.3M) | 4x (8M) | 10x (33M) |
|----------|--------------|---------|-----------|
| lcg_glibc | train 100%, test 0.4% | train 0.8%, test 0.5% | train 0.8%, test 0.5% |
| lfsr_31 | train 100%, test 50% | train 100%, test 50% | train 0.8%, test 0.5% |

The 10x model's LFSR-31 reached 50% test accuracy early in training (step ~500), but collapsed to 0.5% as training continued. Initial hypothesis: weight decay = 0.5 was too strong, decaying the fragile learned structure—"learned then forgotten."

**Results (Phase 4: Reducing weight decay)**:

To test this hypothesis, we reduced the 10x model's weight decay from 0.5 to 0.01 (50x reduction), keeping all other parameters identical:

| Sequence | 10x (wd=0.5) | 10x (wd=0.01) |
|----------|-------------|---------------|
| lcg_glibc | train 0.8%, test 0.5% | train 0.8%, test 0.5% |
| lfsr_31 | train 0.8%, test 0.5% | **train 0.8%, test 0.5%** |

**Hypothesis rejected.** Reducing weight decay by 50x produces identical results.

**Key Findings**:

1. **lcg_glibc**: Larger models did not improve generalization; neither model scale nor regularization strength matters

2. **LFSR-31 collapse is not caused by regularization**: The 10x model cannot stably maintain 50% partial Grokking regardless of weight decay. The problem lies in the instability of the optimization landscape—the parameter space is too large for the fragile 7-bit learning structure to persist

3. **Small model is optimal**: The 0.3M model can stably rest on the 50% saddle point; the larger model slides off

4. **Learnability boundary is a joint property of task and model**: Not only determined by task complexity; the effect of model scale can be counter-intuitive—larger models more easily lose fragile partial learning

**Conclusion**: Scaling up models cannot break the learnability boundary and may actually make things worse. This contradicts the general intuition of scaling laws, but near the learnability boundary, the mathematical structure of the task matters more than model capacity.

> **"It's not that the model is too small; it's that the pattern is too complex. The larger model not only didn't help—it lost the 7 bits it had learned."**

### 6.3 Context Window Experiment: Information Bottleneck Is the Real Cause

Phases 2-4 proved that scaling up models cannot break the 50% ceiling. But this leaves a question: is the ceiling due to insufficient model capacity, or insufficient input information?

**Information analysis**: LFSR-31 outputs 8 bits per step, with 7-bit overlap between adjacent steps (since LFSR shifts by 1 bit per step). At context_len=16, independent information = 8 + 15×1 = 23 bits < 31-bit internal state. Even with perfect reasoning ability, the model cannot reconstruct 31-bit state from 23 bits of information—**a hard information-theoretic limit**.

At context_len=32, independent information = 8 + 31×1 = 39 bits > 31 bits, theoretically sufficient to reconstruct the full internal state.

**Results**:

| Configuration | context_len | Independent Information | Test Accuracy |
|--------------|-------------|------------------------|---------------|
| Small model (0.3M) | 16 | ~24 bits < 31 bits | 50% |
| Small model (0.3M) | **32** | ~39 bits > 31 bits | **99.8%** |
| Small model (0.3M) | **64** | ~71 bits >> 31 bits | **100%** |

**Hypothesis perfectly validated.** Once the window provides sufficient information, the 50% ceiling vanishes instantly. The pattern redundancy at context_len=64 further eliminates the 0.2% residual error from context_len=32—information shows a saturation effect, but additional redundancy still provides marginal benefit.

This result forms a perfect closed loop with Phases 2-4:
- Scale up model (0.3M → 33M): Ineffective, even harmful
- Reduce regularization (wd 0.5 → 0.01): Ineffective
- **Expand window (16 → 32 → 64): 50% → 99.8% → 100%, directly solved**

**Conclusion**: LFSR-31's 50% ceiling is not a model capacity problem—it is an information bottleneck. What was needed was more information, not more parameters.

> **"It's not that the model is too dumb, nor that the pattern is too hard—you simply didn't give it enough clues."**

### 6.4 Data Scaling Experiment: Topological Shredding Is Irreversible

Phase 5 proved that LFSR-31's 50% ceiling can be broken by expanding the context window—because the problem was insufficient information. But what about lcg_glibc's unlearnability? Is it insufficient data, or does no learnable manifold exist?

We expanded lcg_glibc's sequence length from 10,000 to 200,000 (training samples from ~3,000 to ~60,000), keeping all other parameters identical:

| Configuration | seq_length | Training Samples | Train Acc | Test Acc |
|--------------|-----------|-----------------|-----------|----------|
| Original (0.3M) | 10,000 | ~3,000 | 100% | 0.4% |
| 20x data (0.3M) | 200,000 | ~60,000 | **22%** | **0.3%** |

**More data actually made things worse**—the original 3,000 samples could be memorized (train 100%), but 60,000 exceeds the model's memorization capacity (train 22%). Test accuracy remains at random baseline throughout.

This result reveals the truth behind Phase 1's lcg_glibc "100% train + 0.4% test": that 100% training accuracy was not "learning the pattern"—it was 3,000 samples fitting within the model's memorization capacity. When data exceeds this capacity (60,000 >> memorization limit), even memorization fails.

**The contrast with LFSR forms a perfect closed loop**:
- **LFSR-31**: Expand window (more information) → 50% → 100%. The problem was insufficient information; the manifold exists but was incompletely observed
- **lcg_glibc**: Expand data (more samples) → still 0%. The problem is no learnable manifold exists; no amount of data can create structure where there is none

> **"More data won't help—it's not that the clues are insufficient; the answer simply doesn't exist."**

### 6.5 Other Limitations

1. **No fine-grained exploration**: 256 and $2^{31}$ differ by 23 orders of magnitude; where exactly is the phase transition point?

2. **Which bit can't LFSR-31 learn**: The specific bit that remains unlearnable has not been verified

### 6.6 Future Directions

1. **Fine-grained phase transition curve**: Test LCG with state spaces $2^{10}$, $2^{15}$, $2^{20}$ to precisely locate the phase transition point

2. **Dissecting LFSR-31**: Use Mechanistic Interpretability to find which bit the model cannot learn

3. **Mechanism of large model collapse**: Why does the 10x model reach 50% early in training then collapse? Is there a critical parameter count beyond which partial Grokking becomes unsustainable?

---

## 7. Conclusion

This paper experimentally validates the core prediction of the Epiplexity paper: **Information is observer-dependent**.

The same pseudo-random sequence:
- For a cryptographer (who knows the algorithm): Epiplexity $\approx$ seed length
- For a 2-layer Transformer: Epiplexity may be 0 (completely unlearnable)

We found the boundary of neural network learnability:
- **State space $\leq 256$**: Learnable (100% test accuracy)
- **State space = 2³¹**: Unlearnable (test accuracy = random)
- **LFSR-31**: Partially learnable (50% = learned 7/8 bit)

Model scale ablation experiments further reveal:
- **Scaling up cannot break the learnability boundary**: 4x and 10x models are both ineffective on lcg_glibc
- **Larger models perform worse**: The 10x model loses the 7-bit partial Grokking that the small model achieved
- **Not caused by regularization**: Reducing weight decay by 50x cannot rescue the collapse

Context window expansion reveals the true cause of the 50% ceiling:
- **Information bottleneck**: Independent information at context_len=16 (~24 bits) is insufficient to reconstruct the 31-bit internal state
- **Window expansion directly solves it**: context_len 16→32→64, test accuracy 50% → 99.8% → 100%
- **Information saturation effect**: context_len=32 already breaks the ceiling; the extra redundancy at 64 eliminates the last 0.2% residual
- **What was needed was information, not parameters**: Scaling up models was ineffective; expanding the window directly solved the problem

Data scaling experiment validates the root cause of LCG's unlearnability:
- **Topological shredding is irreversible**: lcg_glibc data expanded 20x, test accuracy unchanged (0.3%), training accuracy dropped from 100% to 22%
- **When no manifold exists, data is meaningless**: More data only exposed that the original "100% train" was an illusion (pure memorization)

Implications for Scaling Laws:

Current mainstream scaling laws (Chinchilla, etc.) assume performance = $f$(parameters, data, compute), with optimal allocation among the three. Our experiments reveal three blind spots in this framework:

- **Unstated prerequisite**: The power-law relationships in scaling laws hold only when the data contains a learnable manifold. When the manifold is topologically shredded (as with LCG), all three axes fail simultaneously—20x data + 100x parameters = 0% generalization.
- **Information density as a fourth axis**: The current framework conflates two distinct dimensions under "data": the number of samples and the information density per sample. The LFSR-31 experiment shows that 2x window (information density) outperforms 100x parameters—information density is a fourth scaling dimension independent of data volume.
- **Inverse scaling regions exist**: Current scaling laws predict monotonically increasing performance with scale. We observe non-monotonic behavior near the learnability boundary—the 10x model loses the 7 bits that the small model learned. This suggests scaling laws may require correction in boundary regions.

**One sentence summary**: Computational power determines what you can see in the data—but even more important than computational power is how many clues you get to see, and whether learnable structure exists behind those clues.

---

## References

1. Finzi, M., Qiu, S., Jiang, Y., Izmailov, P., Kolter, J. Z., & Wilson, A. G. (2026). From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence. arXiv:2601.03220.

2. Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.

3. Power, A., et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177.

4. Jin, Y., & Zhao, L. (2026). Grokking as Manifold Discovery: A Geometric Reinterpretation of Delayed Generalization. Zenodo. DOI:10.5281/zenodo.18388631.

---

*"Shannon asked 'how much information does this data objectively contain'; Epiplexity asks 'given your capabilities, what can you learn from it.'"*
