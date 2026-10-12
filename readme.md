# Rap Lyric Generation SLM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karnavagarwal/rap-lyric-slm/blob/main/demo.ipynb)
![Parameters](https://img.shields.io/badge/Parameters-131M-blue)
![Tokens](https://img.shields.io/badge/Tokens-700M%2B-orange)
![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-red)

A 131M-parameter decoder-only Small Language Model (SLM) trained from scratch in PyTorch for multi-syllable rap lyric generation. The model implements a GPT-2 style Transformer architecture trained on over 700M tokens extracted from a curated corpus of 1.7M rap songs.

---

## Model Specifications

| Parameter | Specification |
| :--- | :--- |
| **Architecture** | GPT-2 Style Decoder-Only Transformer |
| **Total Parameters** | 131,922,432 (~131M) |
| **Layers / Transformer Blocks** | 12 |
| **Hidden Dimension ($d_{\text{model}}$)** | 768 |
| **Attention Heads** | 12 (64 dimensions per head) |
| **Context Window ($T$)** | 1024 tokens |
| **Activation Function** | GELU |
| **Normalization** | Pre-Layer Normalization (`LayerNorm`) with residual connections |
| **Vocabulary Size** | 30,000 (Custom BPE) |
| **Training Steps** | 128,000 steps |

---

## Technical Pipeline

### 1. Data Ingestion & Preprocessing
- **Source Corpus:** Extracted from the [Genius Song Lyrics Dataset](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information) on Kaggle (~5.1M songs).
- **Filtering:** Isolated roughly 1.7M English rap songs based on genre tags and metadata.
- **Normalization:** Lyrics were cleaned, stripped of corrupted metadata/non-lyrical tags, and consolidated into a standardized 2.1 GB CSV file (`rap_lyrics_normalized.csv`).

### 2. Tokenization
- Trained a custom Byte-Pair Encoding (BPE) tokenizer from scratch (`train_tokenizer.py`) using the Hugging Face `tokenizers` library.
- Capped vocabulary size at **30,000 tokens** to balance sequence compression against subword granularity, preserving slang, cadence markers, and rhyme patterns.
- Serialized merge rules and vocab directly to `rap_tokenizer.json`.

### 3. High-Throughput Binary Serialization
- Converted the entire 700M+ token corpus into raw unsigned 16-bit integers (`uint16`) using `prepare_data.py`, writing to a flat binary file (`train_data.bin`).
- Training utilizes NumPy memory mapping (`np.memmap`) to stream batches directly from disk, completely bypassing host RAM bottlenecks and enabling near-zero I/O overhead.

### 4. Architecture Implementation
- Implemented from scratch in PyTorch (`model.py`):
  - **Causal Self-Attention:** Hardware-efficient projection combining Query, Key, and Value ($QKV$) operations into a single matrix multiplication, masked with a lower-triangular causal attention buffer.
  - **Feed-Forward Blocks:** Two-stage linear expansion ($768 \rightarrow 3072 \rightarrow 768$) with GELU activations.
  - **Pre-LN Structure:** LayerNorm applied prior to attention and MLP blocks, wrapped with residual skip connections to ensure gradient propagation across 12 layers.

### 5. Training & Optimization
- **Infrastructure:** Remote cloud GPU instances (Vast.ai).
- **Schedule:** 128,000 optimization steps utilizing AdamW with weight decay and learning rate warm-up/decay schedules.
- **Checkpointing:** State dicts periodically serialized and validated against loss convergence, with the final weights released under `rap_model_checkpoint.pt`.

### 6. Inference & Interactive Deployment
- **Dynamic Sampling:** Custom sampling loop (`generate.py`) supporting:
  - Scaled temperature tuning.
  - Top-$k$ filtering.
  - Token-level repetition penalty across a rolling context window to prevent generative looping.
- **Streaming Web UI:** Gradio application (`app.py`) providing a token-by-token streaming interface.
- **Zero-Setup Demo:** Integrated via `demo.ipynb` for one-click interactive execution on Google Colab.

---

## Repository Structure

```text
├── app.py                     # Gradio web UI application with streaming
├── generate.py                # Sampling engine (top-k, temperature, repetition penalty)
├── model.py                   # Custom PyTorch GPT-2 architecture
├── rap_tokenizer.json         # 30,000-vocabulary BPE tokenizer config
├── demo.ipynb                 # Interactive Google Colab notebook
└── README.md
```

> **Note:** Model weights (`rap_model_checkpoint.pt`, ~500 MB) are hosted on [GitHub Releases](https://github.com/karnavagarwal/rap-lyric-slm/releases/tag/ModelRelease).

---

## Quickstart

### 1. Interactive Demo (No Local Setup)
Click the badge at the top of the README to launch the pre-configured [Google Colab Notebook](https://github.com/karnavagarwal/rap-lyric-slm/blob/main/demo.ipynb).

### 2. Local Installation & Inference

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/karnavagarwal/rap-lyric-slm.git](https://github.com/karnavagarwal/rap-lyric-slm.git)
   cd rap-lyric-slm
   ```

2. **Install dependencies:**
   ```bash
   pip install torch gradio tokenizers
   ```

3. **Download model weights:**
   Download `rap_model_checkpoint.pt` from the [Model Release page](https://github.com/karnavagarwal/rap-lyric-slm/releases/download/ModelRelease/rap_model_checkpoint.pt) and place it in the project root:
   ```bash
   wget [https://github.com/karnavagarwal/rap-lyric-slm/releases/download/ModelRelease/rap_model_checkpoint.pt](https://github.com/karnavagarwal/rap-lyric-slm/releases/download/ModelRelease/rap_model_checkpoint.pt)
   ```

4. **Launch the web interface:**
   ```bash
   python app.py
   ```

5. **Terminal generation (optional):**
   ```bash
   python generate.py
   ```

---

## Author
**Karnav Agarwal**  
Indian Institute of Technology Gandhinagar (IITGN)