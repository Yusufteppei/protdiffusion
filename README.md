# protdiffusion

A from-scratch PyTorch implementation exploring **geometric diffusion models for protein structure generation**.

`protdiffusion` began as an attempt to understand the machinery behind **RFdiffusion** by implementing its underlying concepts rather than treating existing implementations as black boxes. The project evolved into a broader exploration of diffusion over protein backbone geometry, including rigid-body transformations, rotations on (SO(3)), invariant point attention, and geometric structure updates.

> **Status:** Experimental research / learning implementation. This project is **not a reproduction of the official RFdiffusion implementation** and does not claim comparable performance.

## Motivation

Protein structures are fundamentally geometric objects.

A protein backbone can be represented as a collection of residue-level rigid frames:

[
T_i = (R_i, t_i)
]

where:

* (R_i \in SO(3)) represents the orientation of residue (i)
* (t_i \in \mathbb{R}^3) represents its position

This makes protein generation different from ordinary diffusion over unconstrained Euclidean vectors. Translations live in (\mathbb{R}^3), while orientations live on the nonlinear manifold (SO(3)).

The goal of `protdiffusion` is to explore how generative models can operate directly on these geometric representations while respecting the symmetries and structure of protein backbones.

## What Was Implemented

The repository explores and implements components including:

* Protein backbone representation using residue-level rigid frames
* Rotation and rigid-body geometry
* (SO(3)) and (SE(3)) transformations
* Protein structure parsing and residue-frame construction
* Translation diffusion
* Experimental rotational diffusion
* Diffusion timestep embeddings
* Single-residue and pair representations
* Invariant Point Attention (IPA)
* Geometric trunk / structure updates
* Noise and structure prediction parameterizations
* Geometric loss functions
* PyTorch-based differentiable geometric operations

The components are intentionally modular so that different representations, diffusion processes, and network architectures can be investigated independently.

## Architecture

The implementation follows the general design pattern of modern protein structure networks and geometric diffusion models:

```text
                    Protein sequence
                           │
                           ▼
                    Input Embedding
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Single representation      Pair representation
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                    Noisy structure
                           x_t
                           │
                           ▼
                         Trunk
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
             IPA                    Transitions
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                 Updated geometric
                    representation
                           │
                           ▼
                 Noise / structure
                     prediction
```

This architecture is **inspired by existing protein structure and diffusion literature**, but it should not be interpreted as a strict implementation of RFdiffusion.

## Geometry

Each residue is represented using a rigid transformation consisting of a rotation and translation:

[
T_i =
\begin{bmatrix}
R_i & t_i \
0 & 1
\end{bmatrix}
]

with:

[
R_i \in SO(3), \qquad t_i \in \mathbb{R}^3
]

The project provides geometry abstractions for applying and manipulating these transformations:

```python
x_rotated = rotation.apply(x)
x_transformed = rigid.apply(x)
```

Rather than representing orientation as an unconstrained (3 \times 3) matrix, rotations are manipulated using geometry-aware operations.

The implementation focuses on making operations such as transformation, composition, inversion, and coordinate changes explicit while remaining differentiable through PyTorch autograd.

## Diffusion

The project explores diffusion over both translational and rotational components of protein backbone frames.

### Translation

Translations can be perturbed using the standard Gaussian diffusion formulation:

[
x_t =
\sqrt{\bar{\alpha}_t}x_0 +
\sqrt{1-\bar{\alpha}_t}\epsilon
]

where:

[
\epsilon \sim \mathcal{N}(0,I)
]

The model can then learn a parameterization of the diffusion process, such as the injected noise or the clean structure.

### Rotation

Rotations require a different treatment because:

[
SO(3) \neq \mathbb{R}^3
]

One approach explored in the project is to construct perturbations in the tangent space:

[
\delta\omega \sim \mathcal{N}(0,\sigma_t^2 I)
]

and map them onto the rotation manifold using the exponential map:

\exp([\delta\omega]_\times)R_t
]

Rotational diffusion is considerably more subtle than ordinary Gaussian diffusion. The appropriate noise distribution, schedule, score representation, and terminal distribution are therefore treated as research questions rather than assumed to be solved by the implementation.

## Training

The training pipeline operates on protein structures represented using batched geometric quantities.

At a high level:

```text
Clean protein structure x₀
            │
            ▼
     Geometric diffusion
            │
            ▼
     Noisy structure xₜ
            │
            ▼
    Geometry-aware network
            │
            ▼
 Predicted noise / structure
            │
            ▼
       Geometric loss
```

Development focused primarily on implementing and validating individual components through small-scale experiments.

The repository should **not** be interpreted as containing a fully trained or validated protein-generation model.

## Project Status

🚧 **Experimental / Learning Implementation**

The core purpose of this repository was to understand the mathematical and computational machinery behind geometric protein diffusion.

### Implemented / explored

* [x] Protein backbone geometry
* [x] Rotation representations
* [x] Rigid transformation abstractions
* [x] Protein structure parsing
* [x] Residue-frame construction
* [x] Input embeddings
* [x] Single and pair representations
* [x] Invariant Point Attention
* [x] Geometric trunk and structure updates
* [x] Translation diffusion
* [x] Diffusion timestep embeddings
* [x] Initial noise-prediction components
* [x] Geometric training losses
* [x] Experimental rotational diffusion

### Not completed / not validated

* [ ] Fully stable end-to-end training
* [ ] Complete principled rotational diffusion process
* [ ] Final validated prediction parameterization
* [ ] Complete reverse denoising process
* [ ] End-to-end sampling
* [ ] Protein backbone generation
* [ ] Structural evaluation
* [ ] Benchmarking against existing protein diffusion models

These unfinished components are intentional boundaries of the current implementation rather than claims of missing functionality in the underlying research literature.

## Project Philosophy

`protdiffusion` is primarily a **learning and research implementation**.

The objective was not to reproduce an existing repository line by line. Instead, existing architectures and papers were used as references for understanding the underlying design choices, mathematical structures, and implementation challenges.

The central question was:

> **What is the simplest and most principled way to perform generative diffusion directly over protein geometry?**

This led to exploration of ideas from RFdiffusion, FrameDiff, AlphaFold-style structure modules, invariant point attention, and geometric diffusion.

The repository therefore represents an exploration of the **machinery behind geometric protein generation**, rather than a claim of state-of-the-art protein design capability.

## Inspirations

The project draws inspiration from research on:

* RFdiffusion
* FrameDiff
* AlphaFold
* RoseTTAFold
* Invariant Point Attention
* Diffusion on (SO(3)) and (SE(3))
* Equivariant neural networks
* Geometric deep learning

## Scope

This repository focuses on understanding and implementing the core geometric and generative machinery required for protein structure diffusion.

It is **not** intended to provide:

* A production-ready protein design system
* Pretrained models
* A drop-in replacement for RFdiffusion
* State-of-the-art protein generation performance
* A reproduction of the published RFdiffusion results

The emphasis is on implementing the underlying concepts from scratch and using them as a platform for further experimentation.

## References

Relevant research includes:

* **RFdiffusion** — protein structure generation through diffusion
* **FrameDiff** — diffusion models operating over protein backbone frames
* **AlphaFold** — geometric structure prediction and structure modules
* **RoseTTAFold** — protein structure prediction
* **Invariant Point Attention** — geometry-aware attention over protein structures
* Research on diffusion processes over (SO(3)) and (SE(3))
* Research on equivariant neural networks and geometric deep learning

More detailed references can be added alongside individual implementations.

## License

License information will be added.
