# Fine-Tuning Llama-3.2-1B Model with LoRA

## Overview
This project demonstrates the process of fine-tuning the `meta-llama/Llama-3.2-1B` model using LoRA (Low-Rank Adaptation) techniques. The training is conducted on a dataset hosted on Hugging Face and optimized for a T4 processor in Google Colab.

## Dataset
The dataset used for training is hosted at [mastermindankur/results](https://huggingface.co/mastermindankur/results). It contains various text samples suitable for the training objectives of the Llama model.

## Requirements
- Python 3.x
- PyTorch
- Hugging Face Transformers
- LoRA implementation
- Google Colab (recommended for training on T4)

## Memory Requirement of a LLM

To calculate the memory requirement for fine-tuning a Large Language Model (LLM), we need to account for the following main components: model parameters, activations, gradients, and optimizer states. Here’s a formula you can use, with details on each component to help you adjust it for your specific model size, precision, and batch size.

Memory Requirement Formula
### Memory Usage in Deep Learning

The total memory requirement \( M_total \) in deep learning can be approximated as:

**M_total = M_parameters + M_activations + M_gradients + M_optimizer**

Where:
- **M_parameters**: Memory required to store model parameters.
- **M_activations**: Memory required for activations during forward pass.
- **M_gradients**: Memory required for gradients during backpropagation.
- **M_optimizer**: Memory used by the optimizer.

## Memory Usage in Deep Learning Models

In deep learning, total memory usage can be calculated as follows:

**M_total = M_parameters + M_activations + M_gradients + M_optimizer**

### Breaking Down Each Component

#### 1. Memory for Model Parameters (M_parameters)

- **Formula**:  
  **M_parameters = num_parameters × bytes_per_parameter**

- **Explanation**:
  - **num_parameters**: Total number of parameters in the model.
  - **bytes_per_parameter**: Memory per parameter, typically 2 bytes for FP16 (mixed precision) or 4 bytes for FP32 (full precision).

#### 2. Memory for Activations (M_activations)

- **Formula**:  
  **M_activations = batch_size × seq_length × hidden_size × num_layers × bytes_per_activation**

- **Explanation**:
  - **batch_size**: Number of samples processed in one forward/backward pass.
  - **seq_length**: Number of tokens in each sample (e.g., 512, 1024, or 2048).
  - **hidden_size**: Dimensionality of the model’s hidden layers.
  - **num_layers**: Total number of layers in the model.
  - **bytes_per_activation**: Memory per activation, typically 2 bytes for FP16 or 4 bytes for FP32.
  - **Note**: Activations are stored temporarily during training to compute gradients in the backward pass.

#### 3. Memory for Gradients (M_gradients)

- **Formula**:  
  **M_gradients = M_parameters**

- **Explanation**:
  - Gradients for each parameter are stored in memory during training, so the memory needed for gradients is roughly equivalent to the memory for model parameters.

#### 4. Memory for Optimizer States (M_optimizer)

- **Formula**:  
  **M_optimizer = num_parameters × bytes_per_optimizer_state × num_optimizer_states**

- **Explanation**:
  - **bytes_per_optimizer_state**: Memory per state variable, typically 2 bytes for FP16 or 4 bytes for FP32.
  - **num_optimizer_states**: Number of states required by the optimizer (Adam/AdamW uses 2 states per parameter: mean and variance).
  - **Example**: For Adam/AdamW, `num_optimizer_states` is 2, so the optimizer memory is roughly twice the model parameter memory.
