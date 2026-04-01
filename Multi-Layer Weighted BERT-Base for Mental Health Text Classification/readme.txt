MULTI-LAYER WEIGHTED BERT FOR MENTAL HEALTH TEXT CLASSIFICATION

PROJECT OVERVIEW
This project implements a high-performance text classification model designed to identify mental health indicators in social media posts. The system categorizes text into four distinct states: Normal, Depression, Anxiety, and Suicidal. Unlike standard models that only use the final output of a transformer, this architecture utilizes a specialized weighting system to capture deep contextual nuances.

MODEL ARCHITECTURE: WEIGHTED BERT-BASE
The core engine is based on the 'bert-base-uncased' transformer model. The architecture includes several custom modifications:

Learnable Layer Weights: The model tracks 13 individual layers, including the initial embedding layer and the 12 subsequent encoder blocks.

Softmax Normalization: A softmax function is applied to these 13 weights to dynamically determine which layers provide the most relevant information for classification.

Weighted Feature Fusion: Instead of relying on a single output, the model calculates a weighted sum across all hidden states.

Classification Head: The fused representation is processed through a dropout layer (30% probability) and a final linear layer to produce the class prediction.

Differential Learning: The BERT backbone is fine-tuned at a very low learning rate (2e-5), while the custom weighting parameters use a higher rate (1e-3) for faster convergence.

FINAL CLASSIFICATION REPORT (BALANCED TEST SET)
The following results were achieved on a balanced evaluation set of 1,000 samples:

Category,  Precision,Recall,F1-Score,Support
Normal,    0.89,    0.97,    0.93,    250
Depression,0.87,    0.89,    0.88,    250
Anxiety,   0.97,    0.86,    0.91,    250
Suicidal,  0.91,    0.91,    0.91,    250

Overall Accuracy: 91%

Macro Average F1: 0.91

ENVIRONMENT AND DEPENDENCIES
The system is built using Python and the PyTorch framework. The following libraries are required:

1. Transformers (Hugging Face)

2. Datasets

3. Accelerate

4. Scikit-learn

5. Pandas and Numpy

6. Matplotlib and Seaborn (for visualization)

DATASET ACKNOWLEDGEMENT
The training utilized the Mental Health Text Classification Dataset (Unbalanced Pool) for broad training, with a synthesized balanced set for final testing validation. Access to gated datasets was verified via Hugging Face authentication.
