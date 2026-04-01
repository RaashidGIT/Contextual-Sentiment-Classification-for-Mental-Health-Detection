**PROJECT TITLE: INTERACTIVE MULTI-LAYER WEIGHTED BERT-BASE FOR MENTAL HEALTH DETECTION**

**OVERVIEW**
This project represents an advanced Natural Language Processing (NLP) solution for identifying mental health indicators in text. Using a customized BERT-Base architecture, the model classifies input into four categories: Normal, Depression, Anxiety, and Suicidal. The project spans the entire pipeline from custom architectural design and differential training to a live, interactive web deployment.

**SCIENTIFIC ARCHITECTURE: LEARNABLE LAYER WEIGHTING**
Most transformer-based models rely only on the final layer for classification. This project implements a "Weighted BERT" approach (MSc Level Evidence):
* **Hidden State Fusion**: The model extracts hidden states from all 13 layers of the BERT-Base-Uncached backbone (1 embedding layer + 12 encoder blocks).
* **Learnable Parameters**: A set of 13 learnable weights is introduced. Through a Softmax layer, the model "learns" which layers provide the most significant contextual clues for mental health sentiment.
* **Differential Learning Rates**: To optimize performance, the BERT backbone is fine-tuned at a lower rate (2e-5) while the weighting parameters are trained at a higher rate (1e-3).

**FINAL PERFORMANCE METRICS**
Evaluation conducted on a "Golden" Balanced Test Set (250 samples per class):

Category        | Precision | Recall | F1-Score | Support
----------------|-----------|--------|----------|--------
Normal          |   0.89    |  0.99  |   0.94   |   250
Depression      |   0.94    |  0.93  |   0.93   |   250
Anxiety         |   0.99    |  0.86  |   0.92   |   250
Suicidal        |   0.93    |  0.96  |   0.95   |   250

* **Overall Accuracy**: 93%
* **Macro Average F1-Score**: 0.93

**INTERACTIVE DEPLOYMENT (GRADIO)**
The project includes a Module 8 deployment using the Gradio framework. This creates a public web interface that allows users to:
1. Input custom text or social media post samples.
2. Receive real-time classification results.
3. View a breakdown of "Confidence Scores" for each of the four mental health categories.

**TECHNICAL REQUIREMENTS**
The system is implemented in Python using the following frameworks:
* **PyTorch**: Deep learning backend and custom model architecture.
* **Hugging Face Transformers**: BERT backbone and Tokenization.
* **Gradio**: Web interface and model hosting.
* **Seaborn/Matplotlib**: Visualization of the learned layer importance.

**DATASET CREDITS**
* Primary Dataset: Mental Health Text Classification Dataset (Kaggle).
* Verification Dataset: AIMH/SWMH (Hugging Face).
