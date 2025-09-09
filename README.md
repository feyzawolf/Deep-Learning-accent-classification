# Accent Classification with Deep Learning

This repository contains the code for the **Deep Learning project** at Tilburg University.  
The task was to classify **speaker accents** from audio recordings using deep learning models.

---

## Project Overview

- **Goal**: Compare different architectures for accent classification from audio.  
- **Dataset**: 3,166 `.wav` files (4 seconds each, resampled to 16 kHz).  
- **Input types**:  
  - Raw audio waveforms (for RNN).  
  - Mel spectrograms (for Transformer).  
- **Models compared**:  
  - 2-layer bidirectional GRU (RNN).  
  - Transformer encoder with self-attention.  
- **Techniques used**:  
  - Data augmentation (time/frequency masking, noise, time-shifting).  
  - Normalization and zero-padding.  
  - Hyperparameter tuning (hidden sizes, attention heads, learning rate, weight decay).  

---

## Results

| Model                                | Input Type       | Augmentation | Validation Accuracy | Macro F1 |
|--------------------------------------|-----------------|--------------|---------------------|----------|
| GRU (2-layer, 128)                   | Raw waveforms   | No           | ~65%                | 0.6285   |
| GRU (2-layer, 128)                   | Raw waveforms   | Yes          | ~62%                | 0.6249   |
| Transformer (256-d, 4 layers, 8 heads) | Mel spectrograms | No          | **95.6%**           | **0.8635** |
| Transformer (256-d, 4 layers, 8 heads) | Mel spectrograms | Yes         | ~95%                | 0.8402   |

- The **Transformer with Mel spectrograms** clearly outperformed the GRU.  
- Augmentation improved robustness and reduced gender bias.  
- Some accents were more challenging (lowest per-class F1 ≈ 0.74).  

---

## Implementation

- **Preprocessing**  
  - Resample audio to 16kHz, pad/truncate to 4 seconds.  
  - Compute 64-band Mel spectrograms (1024 FFT, hop 512).  
  - Convert to dB scale, pad to fixed size.  

- **Models**  
  - **GRU**: 2 bidirectional layers, hidden size 128, dropout 0.3.  
  - **Transformer**: 4 encoder layers, 8 attention heads, 256 hidden dims, learnable positional encodings.  

- **Training**  
  - 100 epochs, Adam optimizer (`lr=1e-4`, `weight_decay=1e-6`).  
  - Augmentation applied differently for GRU vs Transformer.  

---
## Key Takeaways

- Transformers are highly effective for **accent classification** when combined with **Mel spectrogram input**.  
- RNNs (GRUs) underperformed on raw waveforms due to difficulty modeling long sequences.  
- Data augmentation supports robustness, especially for higher-capacity models.  
- Future work could explore **self-supervised models** like Wav2Vec for improved performance.  

