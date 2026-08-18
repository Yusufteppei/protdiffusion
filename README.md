# prot-diffusion

A research-oriented implementation of **geometric diffusion models for protein structure generation**.

This project began as an implementation/reconstruction of ideas from **RFdiffusion**, with the goal of understanding the underlying machinery rather than treating the model as a black box. It is now evolving toward experimenting with alternative formulations of protein diffusion, particularly diffusion directly over protein backbone geometry.

## Motivation

Protein structures are naturally geometric objects. A backbone residue can be represented by a rigid transformation:

$$
T_i = (R_i, t_i)
$$

where:

* $R_i \in SO(3)$ represents orientation
* $t_i \in \mathbb{R}^3$ represents translation

This makes protein generation fundamentally different from ordinary diffusion over Euclidean vectors or images.

The goal of `prot-diffusion` is to explore how diffusion models can operate directly on these geometric representations while preserving the relevant symmetries and structure of proteins.

## Current Direction

The project currently explores:

* Protein backbone representation using rigid frames
* Rotation and translation geometry
* Forward diffusion of protein structures
* Diffusion timestep embeddings
* Residue and pair representations
* Invariant Point Attention (IPA)
* $SE(3)$ / $SO(3)$ geometric diffusion
* Noise- and structure-prediction parameterizations
* Alternative rotational noise processes
* Equivariant protein structure generation

The implementation is intentionally modular so that different diffusion processes and network architectures can be experimented with independently.

## Architecture

The current architecture is inspired by modern protein structure models and diffusion systems:

```text
Protein sequence
       │
       ▼
 Input Embedding
       │
       ├───────────────┐
       │               │
       ▼               ▼
   Single            Pair
representation   representation
       │               │
       └───────┬───────┘
               │
               ▼
          Diffusion
          timestep
               │
               ▼
             Trunk
               │
        ┌──────┴──────┐
        │             │
       IPA        Transitions
        │             │
        └──────┬──────┘
               │
               ▼
      Structure prediction
               │
               ▼
       Denoised protein
```

The architecture is still under active development and is not intended to be a strict reproduction of RFdiffusion.

## Geometry

Backbone residues are represented using rigid transformations:

$$
T_i =
\begin{bmatrix}
R_i & t_i \
0 & 1
\end{bmatrix}
$$

with rotations represented on $SO(3)$.

This allows operations such as:

```python
x_rotated = rotation.apply(x)
x_transformed = rigid.apply(x)
```

rather than treating orientations as unconstrained matrices.

## Diffusion

The project is investigating diffusion processes over both translational and rotational components.

For translations, the standard Gaussian formulation is straightforward:

$$
x_t =
\sqrt{\bar{\alpha}_t}x_0 +
\sqrt{1-\bar{\alpha}_t}\epsilon
$$

with

$$
\epsilon \sim \mathcal{N}(0,I).
$$

Rotational diffusion requires additional geometric treatment because rotations live on $SO(3)$ rather than $\mathbb{R}^3$.

One direction being investigated is representing small rotational perturbations in the tangent space:

$$
\delta\omega \sim \mathcal{N}(0,\sigma_t^2 I)
$$

and mapping them onto $SO(3)$ through the exponential map:

$$R_{t+1} = \exp([\delta\omega]_\times)R_t$$

The appropriate noise schedule and terminal distribution are subjects of experimentation.

## Inspirations

The project draws heavily from existing work in protein structure prediction and geometric diffusion, including:

* RFdiffusion
* FrameDiff
* AlphaFold / RoseTTAFold-style geometric representations
* Diffusion models on $SO(3)$ and $SE(3)$
* Invariant Point Attention
* Equivariant neural networks

The objective is not to reproduce any single model indefinitely, but to understand these methods deeply enough to experiment with alternative formulations.

## Status

🚧 **Research / experimental**

The implementation is incomplete and APIs are expected to change.

Current work is focused on:

* [ ] Complete geometric diffusion process
* [ ] Rotational diffusion
* [ ] Noise/structure prediction formulation
* [ ] Training objective
* [ ] Full denoising sampler
* [ ] Protein backbone generation
* [ ] Structural evaluation
* [ ] Comparison against existing protein diffusion approaches

## Project Philosophy

`prot-diffusion` is primarily a **learning and research project**.

Rather than simply reproducing an existing architecture, the aim is to understand:

> **What is the simplest and most principled way to perform generative diffusion directly over protein geometry?**

Existing methods are therefore treated as starting points rather than fixed constraints.

## References

Key references include:

* RFdiffusion
* FrameDiff
* Diffusion models on $SO(3)$
* AlphaFold
* Invariant Point Attention

More detailed references will be added as individual components are implemented.

## License

License information will be added as the project matures.
