# protdiffusion


A research-oriented PyTorch implementation exploring **geometric diffusion models for protein structure generation**.

The project began as a from-scratch reconstruction of ideas from **RFdiffusion**, with the goal of understanding and implementing the underlying machinery rather than treating existing models as black boxes. It is now evolving into a more experimental project focused on diffusion directly over protein backbone geometry.

> **Status:** Experimental and under active development. This is not a reproduction of the official RFdiffusion implementation.

## Motivation

Protein structures are fundamentally geometric objects.

A protein backbone can be represented as a collection of residue-level rigid frames:

$$
T_i = (R_i, t_i)
$$

where:

* $R_i \in SO(3)$ represents the orientation of residue $i$
* $t_i \in \mathbb{R}^3$ represents its position

This makes protein generation different from ordinary diffusion over unconstrained Euclidean vectors or images. Rotations live on the nonlinear manifold $SO(3)$, while translations live in $\mathbb{R}^3$.

The goal of `protdiffusion` is to explore how diffusion models can operate directly on these geometric representations while preserving the symmetries and structure relevant to proteins.

## Current Direction

The project currently includes or explores:

* Protein backbone representation using residue-level rigid frames
* Rotation and rigid-body geometry
* $SO(3)$ and $SE(3)$ transformations
* Translation diffusion
* Experimental rotational diffusion
* Diffusion timestep embeddings
* Residue and pair representations
* Invariant Point Attention (IPA)
* Structure and backbone updates
* Noise prediction parameterizations
* Geometric loss functions
* Equivariant protein structure generation

The implementation is intentionally modular so that different geometric representations, diffusion processes, and network architectures can be experimented with independently.

## Architecture

The current architecture follows the general structure of modern protein geometry networks and diffusion models:

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
                           │
                           ▼
                    Denoised structure
```

The exact architecture is still evolving and is **not intended to be a strict reproduction of RFdiffusion**.

## Geometry

Each residue is represented using a rigid transformation consisting of a rotation and translation:

$$
T_i =
\begin{bmatrix}
R_i & t_i \
0 & 1
\end{bmatrix}
$$

where:

$$
R_i \in SO(3)
$$

and:

$$
t_i \in \mathbb{R}^3
$$

The project implements geometric abstractions for working directly with these transformations:

```python
x_rotated = rotation.apply(x)
x_transformed = rigid.apply(x)
```

Rather than treating orientation as an unconstrained $3 \times 3$ matrix, rotations are represented and manipulated using geometry-aware operations.

The goal is to make transformations, composition, inversion, and coordinate changes explicit while keeping the underlying implementation compatible with PyTorch autograd.

## Diffusion

The project investigates diffusion over both translational and rotational components of protein backbone frames.

### Translation

Translations can use the standard Gaussian diffusion formulation:

$$
x_t =
\sqrt{\bar{\alpha}_t}x_0 +
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

where:

$$
\epsilon \sim \mathcal{N}(0, I)
$$

The network can then be trained to predict a parameterization of the diffusion process, such as the injected noise or the clean structure.

### Rotation

Rotations require additional treatment because:

$$
SO(3) \neq \mathbb{R}^3
$$

A direction currently being explored is sampling perturbations in the tangent space:

$$
\delta\omega \sim \mathcal{N}(0, \sigma_t^2 I)
$$

and mapping them onto the rotation manifold through the exponential map:

$$
R_{t+1}
=======

\exp([\delta\omega]_\times)R_t
$$

The appropriate rotational noise process, schedule, target parameterization, and terminal distribution remain active areas of experimentation.

## Training

The current training pipeline operates on protein structures represented as batched rigid frames.

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

The implementation is currently being validated through small-scale overfitting experiments before scaling to larger protein datasets.

## Project Status

🚧 **Research / Experimental**

The repository is under active development. APIs, representations, and training objectives may change significantly.

Current work includes:

* [x] Protein and backbone geometry
* [x] Rotation and rigid transformation abstractions
* [x] Protein structure parsing and residue frames
* [x] Input embedding
* [x] Residue and pair representations
* [x] Invariant Point Attention
* [x] Geometric trunk and structure updates
* [x] Translation diffusion
* [x] Diffusion timestep embeddings
* [x] Initial noise prediction network
* [x] Geometric training loss
* [ ] Stable end-to-end overfitting and training
* [ ] Principled rotational diffusion on $SO(3)$
* [ ] Final noise/structure prediction parameterization
* [ ] Full reverse denoising process
* [ ] Sampling pipeline
* [ ] Protein backbone generation
* [ ] Structural evaluation
* [ ] Comparison with existing geometric protein diffusion approaches

## Project Philosophy

`protdiffusion` is primarily a **learning and research implementation**.

The objective is not simply to reproduce an existing repository layer by layer. Existing architectures are used to understand the design space, after which individual components can be modified or replaced.

The central question motivating the project is:

> **What is the simplest and most principled way to perform generative diffusion directly over protein geometry?**

This means that RFdiffusion, FrameDiff, AlphaFold-style structure modules, and geometric diffusion methods are treated as starting points for investigation rather than fixed architectural constraints.

## Inspirations

The project draws inspiration from work on:

* RFdiffusion
* FrameDiff
* AlphaFold
* RoseTTAFold
* Invariant Point Attention
* Diffusion on $SO(3)$ and $SE(3)$
* Equivariant neural networks
* Geometric deep learning

## Scope

This repository is currently focused on **understanding and implementing the core geometric and generative machinery**.

It is not currently intended to provide:

* A production-ready protein design system
* Pretrained models
* A drop-in replacement for RFdiffusion
* State-of-the-art protein generation performance

The focus is on building the underlying components from scratch and using them as a platform for experimentation.

## References

Key references include:

* RFdiffusion
* FrameDiff
* AlphaFold
* RoseTTAFold
* Invariant Point Attention
* Diffusion models on $SO(3)$ and $SE(3)$

More detailed paper references and implementation notes will be added as the corresponding components mature.

## License

License information will be added.
