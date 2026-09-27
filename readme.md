# Rap Lyric Generation SLM

## Overview
This repository contains the architecture and training pipeline for a Small Language Model (SLM) designed for rap lyric generation. Built entirely from scratch in PyTorch, the model implements a GPT-2 style decoder-only Transformer architecture.

**Author:** Karnav Agarwal

## Project Workflow & Pipeline

**1. Data Acquisition & Preprocessing**
* **1.a.** The raw data for this project was downloaded from Kaggle: [Link to be inserted].
* **1.b.** This data was then cleaned through basic Python scripts, which firstly isolated the rap songs, and then extracted only the lyrics column.
* **1.c.** The resulting cleaned text was aggregated and normalized into a single 2.1GB CSV file (`rap_lyrics_normalized.csv`).

**2. Tokenization Strategy**
* **2.a.** A custom Byte-Pair Encoding (BPE) tokenizer was trained from scratch on the corpus using a dedicated script (`train_tokenizer.py`).
* **2.b.** The vocabulary size was strictly capped at 30,000 unique tokens to optimally balance compression while preserving the rap-specific language data.
* **2.c.** The final vocabulary and merge rules were exported and saved as `rap_tokenizer.json`.

**3. Data Serialization for Memory Efficiency**
* **3.a.** The entire 2.1GB text corpus was processed through the BPE tokenizer (`prepare_data.py`), yielding a sequence of over 700 million token IDs.
* **3.b.** To bypass system RAM limitations during training, these token IDs were serialized and saved to disk as a flat, 16-bit integer binary file (`train_data.bin`).
* **3.c.** This binary format enables high-throughput, memory-mapped (`np.memmap`) data loading, allowing the hardware to scoop out exact batch chunks dynamically without loading the massive dataset into memory.

**4. Architecture Engineering (Model Construction)**
* **4.a.** A complete GPT-2 style Transformer architecture was coded from scratch in PyTorch (`model.py`), featuring a 1024-token context window and a 768 embedding dimension.
* **4.b.** The network backbone stacks 12 sequential Transformer blocks.
* **4.c.** Each block incorporates Multi-Head Causal Self-Attention (sliced into 12 heads of 64 dimensions, utilizing a persistent lower-triangular causal mask and hardware-optimized QKV generation).
* **4.d.** The attention mechanisms are paired with GELU-activated MLPs. Pre-Layer Normalization and residual bypass connections are integrated throughout to stabilize deep mathematical operations.

**This documents the work done till date.**