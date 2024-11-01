# Transformer Model from Scratch using PyTorch

This repository contains a custom implementation of a Transformer model using PyTorch. The model is designed to perform tasks like machine translation, summarization, or text classification by leveraging self-attention and multi-head attention mechanisms, as outlined in the original ["Attention is All You Need"](https://arxiv.org/pdf/1706.03762) paper.

## Project Overview

This project demonstrates the implementation of a Transformer model from scratch. It includes custom modules for:
- *Embedding layers with positional encoding*
- *Multi-head attention*
- *Encoder and decoder layers*
- *Final linear layer for producing output vocabulary logits*

## Features

- **Self-Attention and Multi-Head Attention**: Each encoder/decoder layer contains self-attention and multi-head attention mechanisms for effective sequence processing
- **Positional Encoding**: Enables the model to incorporate sequence information
- **Fully Customizable**: Number of layers, heads, and feed-forward dimensions can be adjusted

## Model Architecture

The Transformer model consists of:

- **Encoder and Decoder Layers**: Each layer contains a multi-head self-attention mechanism, feed-forward neural network, and normalization layers
- **Positional Encoding**: Adds position information to input embeddings
- **Final Linear Layer**: Projects encoder-decoder outputs to vocabulary logits
[](/image.png)
