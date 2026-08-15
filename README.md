# RFdiffusion — Independent PyTorch Implementation

An independent PyTorch implementation of the core ideas behind **RFdiffusion**, a diffusion-based generative model for protein structure design.

The goal of this project is not simply to use an existing implementation, but to reconstruct the underlying architecture and geometric reasoning from the literature and reference implementations.

> **Status: Work in Progress**

---

## Overview

RFdiffusion combines protein structure representations, invariant geometric reasoning, and iterative denoising to generate protein structures.

This implementation is being built from the architectural components upward, with particular attention to the interaction between:

* Protein sequence representations
* Single and pair representations
* Rigid-body transformations
* Invariant Point Attention (IPA)
* Repeated trunk refinement
* Diffusion-based structure generation

The current model follows the high-level structure:

```text
Input
 │
 ├── Single representation
 └── Pair representation
          │
          ▼
   Input Embedding
          │
          ▼
     ┌───────────┐
     │   Trunk    │
     │            │
     │    IPA     │
     │     ↓      │
     │  Rigid     │
     │  updates   │
     └───────────┘
          │
          ▼
   Repeated refinement
          │
          ▼
 Protein structure
```

The trunk is applied iteratively, allowing the model to progressively refine the structural representation.

---

## Current Implementation

The current top-level model is structured around repeated trunk passes:

```python
class RFDiffusion(nn.Module):
    def __init__(self, trunks):
        super().__init__()

        self.trunks = trunks
        self.input_embedder = InputEmbedder()
        self.trunk = Trunk()

    def forward(self, batch, mask):
        single, pair = self.input_embedder(batch)

        rigids = None

        for _ in range(self.trunks):
            rigids = self.trunk(single, pair, rigids)
```

The important architectural idea here is that the **rigid representation is progressively refined through repeated trunk iterations**.

---

## Components

### Input Embedding

The input embedder produces the model's initial:

* Single representation
* Pair representation

These representations provide the feature space on which the trunk operates.

### Rigid Geometry

Protein structure is represented using rigid transformations consisting of:

[
T = (R, t)
]

where:

* (R) is a 3D rotation
* (t) is a 3D translation

This provides a natural representation for residue-local coordinate frames.

### Invariant Point Attention

IPA provides the geometric attention mechanism used by the trunk.

Unlike conventional attention, IPA combines:

* Scalar attention
* Point-based attention
* Pairwise information

Point queries and keys are transformed through residue rigid frames, allowing geometric relationships to influence attention while maintaining rotational and translational invariance.

### Trunk

The trunk repeatedly processes the single and pair representations while updating the structural rigid representation.

Conceptually:

```text
(single, pair, rigid_0)
          │
          ▼
        Trunk
          │
          ▼
        rigid_1
          │
          ▼
        Trunk
          │
          ▼
        rigid_2
          │
          ▼
          ...
```

This iterative refinement is a central part of the architecture being reconstructed.

---

## Geometry

The implementation includes explicit geometric operations rather than treating protein coordinates as ordinary unconstrained vectors.

Current geometric work includes:

* Rotation representations
* Rigid transformations
* Local ↔ global coordinate transformations
* Protein residue frames
* Backbone geometry
* Dihedral-angle calculations
* Atom placement
* Torsion-based structural reconstruction

The intention is to maintain a clear separation between **learned representations** and **physical/geometric transformations**.

---

## Diffusion

The diffusion component is currently under active development.

The intended generative process is:

```text
Clean protein structure
        │
        ▼
   Forward diffusion
        │
        ▼
     Noisy state
        │
        ▼
   Neural denoiser
        │
        ▼
   Less noisy state
        │
        ▼
      Repeat
        │
        ▼
 Generated structure
```

The final system will use iterative denoising to transform noisy structural states into coherent protein structures.

---

## Design Philosophy

This project is primarily an **implementation and learning exercise**.

Rather than treating RFdiffusion as a black-box model, the implementation focuses on understanding and reconstructing the mechanisms that make geometric protein generation possible.

Particular emphasis is placed on:

1. Understanding the mathematical representation of protein geometry.
2. Implementing rigid-body transformations explicitly.
3. Understanding how IPA combines scalar and geometric attention.
4. Understanding how trunk iterations progressively refine structure.
5. Integrating these components with diffusion-based generation.

---

## Project Status

### Implemented / Under active development

* [x] PyTorch model structure
* [x] Single/pair input embedding interface
* [x] Rotation and rigid-body representations
* [x] Local/global coordinate transformations
* [x] Protein geometric representation
* [x] Backbone geometry
* [x] Dihedral-angle calculations
* [x] Atom placement / structural reconstruction utilities
* [x] Trunk interface
* [ ] Complete IPA validation
* [ ] Complete diffusion process
* [ ] Reverse diffusion / sampling
* [ ] End-to-end structure generation
* [ ] Training objective
* [ ] Quantitative validation against RFdiffusion
* [ ] Generated-structure evaluation

The implementation should therefore **not currently be considered a reproduction of the official RFdiffusion model or a production-ready protein design system**.

---

## References

This project is based primarily on the ideas presented in:

* Watson et al., *De novo design of protein structure and function with RFdiffusion*
* Jumper et al., *Highly accurate protein structure prediction with AlphaFold*
* The RFdiffusion reference implementation

The project is intended as an independent implementation for understanding and experimentation.

---

## Disclaimer

This repository is an independent implementation and is **not affiliated with or endorsed by the original RFdiffusion authors**.

Implementation details may differ from the reference implementation, and the project should not be expected to reproduce its performance until the relevant components have been fully implemented and validated.

