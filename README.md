# MNIST CNN Training Comparison Tool

An interactive web-based tool for training and comparing different configurations of Convolutional Neural Networks (CNNs) on the MNIST dataset. This tool allows real-time visualization of training progress and comparison between different model configurations.

## Features

### Model Architecture
- 4-layer Convolutional Neural Network
- Configurable number of kernels for each layer
- Max pooling after each convolutional layer
- Dropout layers for regularization
- Final fully connected layer
- CUDA support for GPU acceleration

### Interactive Interface
- Split-screen layout for comparing two model configurations
- Real-time visualization of:
  - Training loss
  - Test loss
  - Training accuracy
  - Test accuracy
- Live metric updates during training
- Training history sidebar
- Configurable parameters for each model:
  - Kernel sizes for each layer
  - Batch size
  - Learning rate
  - Optimizer selection (Adam, SGD, RMSprop)
  - Number of epochs

### Training Controls
- Start/Stop training for each configuration
- Reset graphs individually
- Save training results and configurations
- View training history
- Compare different model configurations

### Visualization Features
- Interactive plots using Plotly.js
- Side-by-side comparison of models
- Historical training data viewer
- Downloadable graphs and configurations
- Training logs with timestamps

## Installation

1. Clone the repository: