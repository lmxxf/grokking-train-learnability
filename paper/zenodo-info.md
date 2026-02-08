# Zenodo Upload Information

## Basic Information

**DOI**: 10.5281/zenodo.18524191

**Resource type**: Publication / Preprint

**Title**: Learnability Boundary: How Complex Can Neural Networks Learn Pseudo-Random Sequences?

**Publication date**: 2026-02-07

**Authors/Creators**:
1. Jin, Yanyan (lmxxf@hotmail.com) - ORCID: 0009-0008-0169-0409
2. Zhao, Lei (zhaosanshi@gmail.com) - ORCID: 0009-0008-9765-6837 - Affiliation: Tencent CSIG, Shenzhen, China

**Description**:
```
Classical information theory assumes observers have unlimited computational power, making pseudo-random numbers "low entropy." However, for computationally bounded neural networks, the same data may be completely unlearnable noise. This paper experimentally explores the boundary of neural network learnability: from simple periodic patterns to cryptographically secure random numbers, we identify where Epiplexity (extractable structure) transitions from non-zero to zero.

We train identical Transformer architectures on 6 sequences of varying complexity and find:
(1) Pseudo-random sequences with state space ≤256 (simple LCG, 5-bit LFSR) can be perfectly learned (100% test accuracy);
(2) Pseudo-random with state space 2³¹ (glibc LCG) is completely unlearnable (test accuracy = random guessing);
(3) 31-bit LFSR exhibits partial Grokking—the model learns 1 bit of the sequence's pattern (50% accuracy) but cannot learn the remaining 7 bits.

Model scale ablation (0.3M → 8M → 33M parameters) shows:
- Larger models do NOT break the learnability boundary
- 10x model loses the 1-bit partial Grokking that the small model achieved
- Reducing weight decay by 50x (from 0.5 to 0.01) cannot rescue this collapse
- The problem is optimization landscape instability in larger models, not excessive regularization
- Conclusion: "It's not that the model is too small; it's that the pattern is too complex. The larger model not only didn't help—it lost the only 1 bit it had learned."

These findings validate the core prediction of the Epiplexity paper (Finzi et al., 2026): information is observer-dependent; the same data presents different learnability to observers with different computational power. Furthermore, the learnability boundary is a joint property of task and model, where the effect of model scale can be counter-intuitive.

Key contributions:
- Experimental identification of the learnability phase transition boundary (between state space 256 and 2³¹)
- Discovery of "partial Grokking" phenomenon in LFSR-31 (learning only 1 bit of 8)
- Model scale ablation proving boundary is task property, not capacity limitation
- Weight decay ablation ruling out regularization as cause of large model collapse
- Connection between Epiplexity theory and Grokking manifold discovery

Code and data: https://github.com/lmxxf/grokking-train-learnability
```

**License**: Creative Commons Attribution 4.0 International

---

## Recommended Information

**Keywords**:
- Grokking
- Learnability
- Epiplexity
- Information Theory
- Neural Networks
- Pseudo-random
- Phase Transition
- Transformer
- Manifold Discovery
- Model Scaling
- Weight Decay
- Optimization Landscape
- Underfitting

**Languages**: English

**Version**: 2.0.0

**Publisher**: Zenodo

---

## Related Works

**Related identifiers**:

| Identifier | Relation | Resource type |
|------------|----------|---------------|
| arXiv:2601.03220 | Cites | Publication |
| arXiv:2201.02177 | Cites | Publication |
| 10.5281/zenodo.18388631 | Continues | Publication |
| https://github.com/lmxxf/grokking-train-learnability | IsSupplementedBy | Software |

**References**:
```
Finzi, M., Qiu, S., Jiang, Y., Izmailov, P., Kolter, J. Z., & Wilson, A. G. (2026). From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence. arXiv:2601.03220.

Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.

Power, A., et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. arXiv:2201.02177.

Jin, Y., & Zhao, L. (2026). Grokking as Manifold Discovery: A Geometric Reinterpretation of Delayed Generalization. Zenodo. DOI:10.5281/zenodo.18388631.
```

---

## Software

**Repository URL**: https://github.com/lmxxf/grokking-train-learnability

**Programming language**: Python

**Development Status**: Active
