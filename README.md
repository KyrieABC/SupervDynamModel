# Supervised Dynamics Model

A clean, modular PyTorch project for learning environment transitions with supervised learning:

\[(state, action) \rightarrow next\_state\]

This repository builds a simple but professional machine learning pipeline around a custom 2D point-mass simulator. It includes:

- synthetic dataset generation
- configurable MLP dynamics model
- training with checkpoints
- one-step prediction evaluation
- multi-step rollout comparison
- trajectory visualization

---

## Why this project matters

Learning a forward dynamics model is a foundational problem in:

- model-based reinforcement learning
- robotics
- system identification
- simulation learning
- planning under learned world models

This project demonstrates both ML engineering and software architecture skills in a portfolio-friendly way.

---

## Project architecture

```mermaid
flowchart LR
    A[Simulator] --> B[Dataset Generation]
    B --> C[Train / Val Split]
    C --> D[Normalization]
    D --> E[MLP Dynamics Model]
    E --> F[Training Loop]
    F --> G[Checkpoint]
    G --> H[Evaluation]
    H --> I[One-step Metrics]
    H --> J[Multi-step Rollouts]
    H --> K[Trajectory Plots]
