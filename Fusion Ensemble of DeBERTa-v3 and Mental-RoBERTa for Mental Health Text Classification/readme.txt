================================================================================
PROJECT: Mental Health Text Classification via Transformer Ensemble
MODELS: microsoft/deberta-v3-base & mental/mental-roberta-base
================================================================================

1. OVERVIEW
-----------
This project implements a high-performance mental health detection system 
using an ensemble of state-of-the-art Natural Language Understanding (NLU) 
models. The goal is to classify text into four distinct categories: 
'Normal', 'Depression', 'Anxiety', and 'Suicidal'.

The project utilizes a "Learned Fusion" strategy, combining the generalized 
reasoning power of DeBERTa-v3 with the domain-specific knowledge of 
Mental-RoBERTa (pre-trained on mental health social media data).

2. CORE ARCHITECTURE
--------------------
The notebook defines a custom "MentalHealthClassifier" class that improves 
upon standard [CLS] token pooling:
* Backbone: DeBERTa-v3-base / Mental-RoBERTa.
* Attention-Weighted Pooling: Instead of simple mean pooling, the model 
    applies a learned linear attention layer over all token hidden states 
    to focus on the most emotionally relevant words.
* Concatenation: The [CLS] embedding and the attention-weighted pool are 
    concatenated to create a rich 1536-dimensional representation (for base models).
* Classification Head: A multi-layer perceptron (MLP) with GELU activation, 
    LayerNorm, and Dropout for regularization.

3. DATA PIPELINE
----------------
* Source: Mental Health Text Classification Dataset (Unbalanced).
* Preprocessing: Automatic deduplication and label mapping.
* Zero-Leakage Strategy: Implements a balanced test set extraction logic 
    ensuring that specific sensitive samples are never seen during training.
* Class Imbalance Handling: Calculates and applies class weights during 
    loss computation to prevent model bias toward 'Normal' samples.

4. TRAINING STRATEGY
--------------------
* Optimizer: AdamW with weight decay (0.01).
* Schedules: Cosine Annealing with a 10% warmup ratio.
* Loss Function: Cross-Entropy with Label Smoothing (0.1) and Class Weights.
* Epochs: 6 (Phase 1 for DeBERTa, Phase 2 for Mental-RoBERTa).
* Hardware: Configured for NVIDIA Tesla T4 (Kaggle/Colab).

5. ENSEMBLE METHOD (LEARNED FUSION)
-----------------------------------
The notebook performs a grid search to find the optimal alpha (α) value 
for probability fusion.
* Formula: Final Prediction = α * Prob(DeBERTa) + (1-α) * Prob(Mental-RoBERTa)
* Optimized Alpha: ~0.45 (DeBERTa) and 0.55 (Mental-RoBERTa).
* Performance: Achieved ~89.22% Accuracy and ~88.68% Macro F1-score.

6. EVALUATION METRICS
---------------------
The notebook generates a full suite of evaluation artifacts:
* Per-class accuracy bars (highest performance usually seen in 'Normal').
* Full Classification Report (Precision, Recall, F1).
* Seaborn-based Confusion Matrix visualization (saved as .png).

7. DEPLOYMENT (DEMO)
--------------------
Module 8 integrates the trained weights into a live Gradio web interface.
Users can input text, and the ensemble model provides a real-time 
confidence breakdown across the four categories.

8. REQUIREMENTS
---------------
- Python 3.10+
- PyTorch
- Transformers (v4.48.0 recommended)
- Datasets
- Sentencepiece (required for DeBERTa)
- Gradio (for inference)
- Scikit-learn, Seaborn, Matplotlib

================================================================================
