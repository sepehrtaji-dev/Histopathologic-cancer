# 🔬 Histopathologic Cancer Detection

A deep learning project for **experimental histopathology image classification** using PyTorch and a ResNet18-based neural network.

> ⚠️ This project is strictly educational and experimental. It is not a medical diagnostic system and must not be used for real-world medical decisions.

---

## ✨ Overview

This project explores how a convolutional neural network can learn visual patterns from histopathology images and classify them into two categories:

- 🟢 Non-Cancer
- 🔴 Cancer

The project was built as a hands-on deep learning experiment with a focus on:

- GPU-accelerated training
- Image augmentation
- Transfer learning
- Class balancing
- Validation metrics
- Model checkpointing
- A clean desktop prediction interface

---

## 🖥️ Application

The project includes a modern desktop application built with **PyQt6**, featuring a clean dark interface for loading an image and running the trained model.

<p align="center">
  <img src="assets/dark.png" alt="Histopathologic Cancer Detection - Dark UI" width="900">
</p>

The interface provides:

- 🖼️ Image preview
- 🧠 Neural network prediction
- 📊 Cancer probability
- 🎨 Dark / Light UI
- ⚡ GPU-powered inference
- 🍎 Minimal, modern interface

---

## 🧠 Model

The current model is based on **ResNet18** with a custom binary classification head.

### Architecture

```text
Input Image
    │
    ▼
Resize → 96 × 96
    │
    ▼
ResNet18 Backbone
    │
    ├── Convolutional Layers
    ├── Residual Blocks
    └── Feature Extraction
    │
    ▼
Dropout
    │
    ▼
Linear Layer
    │
    ▼
Cancer Logit
    │
    ▼
Sigmoid
    │
    ▼
Cancer Probability