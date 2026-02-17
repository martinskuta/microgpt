## Overview

This is a single-file C# reimplementation of Andrej Karpathy's **microgpt** Python script. The reimplementation follows
the original Python version almost 1:1, translated into C# because it is my tech stack that I am more familiar with. I
created this version to explore and play with the algorithm for educational purposes.

## About the Original

The original microgpt is a minimal implementation of a GPT-style language model from scratch, created by Andrej
Karpathy. It serves as an educational tool to understand the core mechanics of transformer-based language models.

For detailed information about the algorithm and its implementation, please
visit: https://karpathy.github.io/2026/02/12/microgpt/

## Key Features of the Algorithm

microgpt implements a character-level language model using the Transformer architecture. The key components include:

- **Character-level tokenization**: The model operates directly on characters rather than subword tokens
- **Multi-head self-attention**: Enables the model to focus on different parts of the input sequence simultaneously
- **Position embeddings**: Provides the model with information about the position of tokens in the sequence
- **Feed-forward layers**: Adds non-linear transformations to enhance the model's expressiveness
- **Layer normalization**: Stabilizes training by normalizing activations
- **Residual connections**: Helps with gradient flow during training

The model is trained to predict the next character in a sequence, learning patterns and structure from the training
text.

## C# Implementation Details

This C# version has **no external dependencies** apart from .NET itself. It is a standalone, single-file implementation
that demonstrates how the transformer architecture works without relying on any machine learning frameworks or
libraries.

## Prerequisites

- **.NET 10** must be installed on your system

## How to Run

You can run the program in two ways:

1. **Using the dotnet CLI:**
   ```bash
   dotnet run MicroGPT.cs
   ```

2. **Making the file executable (Unix systems):**
   ```bash
   chmod +x MicroGPT.cs
   ./MicroGPT.cs
   ```

## Credits

Original microgpt by Andrej Karpathy: https://karpathy.github.io/2026/02/12/microgpt/