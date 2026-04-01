==========================================================================
PROJECT: Weighted Ensemble Mental Health Detector
MODELS: Mental-BERT (mental/mental-bert-base-uncased) & DistilRoBERTa
DATASET: Mental Health Text Classification (Balanced & Unbalanced)
==========================================================================

1. OVERVIEW
-----------
This project implements a sophisticated Deep Learning pipeline for classifying 
text into four mental health categories: Normal, Depression, Anxiety, and 
Suicidal. The system uses a "Nuclear Fusion" soft-voting ensemble strategy 
combining two distinct transformer architectures to maximize diagnostic accuracy 
and domain-specific understanding.

2. MODEL ARCHITECTURE
---------------------
The project utilizes a custom "WeightedTransformer" class for both backbones:

- Mental-BERT: A specialized BERT model pre-trained on mental health-related 
  social media data, providing deep domain knowledge.
- DistilRoBERTa: A lightweight, efficient version of RoBERTa that provides 
  structural diversity and helps prevent overfitting.
- Layer Weighting: Instead of using only the final [CLS] token, the model 
  implements a learnable "Layer Weighting" mechanism. It stacks all hidden 
  states from the transformer layers and applies a Softmax-weighted sum to 
  extract the most relevant features across different levels of abstraction.

3. DATA PIPELINE (Zero-Leakage)
------------------------------
- Source: Mental Health Text Classification Dataset.
- Preprocessing: 
    * Duplicate removal and label mapping.
    * Strict "Zero-Leakage" validation: A balanced test set (1000 samples) 
      is extracted before training to ensure evaluation integrity.
    * Stratified splitting for training and validation sets.

4. TRAINING ENGINE
------------------
- Optimizer: AdamW with differential learning rates (Lower for transformer 
  backbones, higher for the custom layer weights).
- Scheduler: Linear schedule with warmup to stabilize initial training.
- Training Process: Sequential training of both models with memory 
  management (GC and CUDA cache clearing) to accommodate Kaggle/GPU 
  constraints.
- Checkpointing: Saves only the "Best" state for each model based on 
  Macro F1 scores.

5. ENSEMBLE EVALUATION & RESULTS
--------------------------------
The system employs a soft-voting ensemble (50/50 fusion of Softmax outputs).
- Overall Accuracy: ~88.28%
- Macro F1 Score: ~87.63%
- Performance Breakdown:
    * Normal: 96.5% Accuracy
    * Anxiety: 90.9% Accuracy
    * Suicidal: 86.4% Accuracy
    * Depression: 78.2% Accuracy

6. DEPLOYMENT & INFERENCE
-------------------------
The notebook includes a Module for real-time interaction:
- Gradio Interface: A web-based UI that allows users to input text and 
  see the ensemble's confidence distribution across categories.
- Exportable Weights: Saves 'best_bert.bin' and 'best_roberta.bin' for 
  production use.

7. REQUIREMENTS
---------------
- PyTorch (2.10.0+cu128)
- Transformers (4.48.0)
- Datasets
- Gradio
- Scikit-learn, Pandas, Seaborn

==========================================================================
