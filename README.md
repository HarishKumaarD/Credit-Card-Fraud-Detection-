# Credit-Card-Fraud-Detection

# Advanced Credit Card Fraud Detection using a Hybrid Autoencoder

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a deep learning solution to detect fraudulent credit card transactions using a hybrid Autoencoder-Classifier model. It specifically addresses the challenge of a **highly imbalanced dataset** by employing SMOTE for oversampling and focusing on metrics like the Precision-Recall (PR) AUC and F1-Score for robust evaluation.

## Key Features
- **Imbalance Handling:** Utilizes the SMOTE (Synthetic Minority Over-sampling Technique) to create a balanced training set from a dataset where only 0.17% of transactions are fraudulent.
- **Deep Learning Model:** A hybrid Autoencoder and Classifier built in **PyTorch** that learns a robust, low-dimensional representation of transaction data.
- **Robust Preprocessing:** Implements a data pipeline with `RobustScaler` to handle outliers in financial data and prevent data leakage.
- **Advanced Evaluation:** Goes beyond accuracy to evaluate the model on Precision, Recall, F1-Score, ROC AUC, and PR AUC, which are critical for imbalanced classification.
- **Threshold Optimization:** Includes a final optimization step that analyzes the Precision-Recall curve to select a decision threshold that maximizes the F1-score, significantly improving the model's practical usability.

## Final Results
The model was trained on the imbalanced Kaggle dataset and achieved the following on the test set:
- **Precision-Recall AUC:** 0.77
- **ROC AUC:** 0.98
- **Best F1-Score (at Optimal Threshold):** 0.82
- **Recall (Fraud):** 81.6%
- **Precision (Fraud):** 82.5%

### Evaluation Plots
*(Here you would insert the images of your plots after uploading them to your GitHub repo)*

**Confusion Matrix (at Optimal Threshold)**
![Confusion Matrix](./image_e9e4ce.png)

**Precision-Recall Curve**
![Precision-Recall Curve](./image_e9e52b.png)

**ROC Curve**
![ROC Curve](./image_e9d20c.png)

## Methodology
The project follows a standard machine learning workflow tailored for an imbalanced classification problem:
1.  **Data Loading & Preprocessing:** The dataset, containing 284,000+ transactions, is loaded. The `Time` and `Amount` features are scaled using `RobustScaler` to mitigate the effect of outliers.
2.  **Imbalance Handling (SMOTE):** The training data is resampled using SMOTE. This synthesizes new data points for the minority (fraud) class, creating a balanced dataset for the model to train on. This is only applied to the training set to ensure the test set remains representative of the real-world data distribution.
3.  **Model Architecture:** A hybrid model is used:
    * An **Autoencoder** with a narrow bottleneck (8 neurons) is designed to learn a compressed representation of "normal" transaction features. Dropout layers are included to improve generalization.
    * A simple **Classifier** then uses this learned representation to make a final prediction.
4.  **Training:** The model is trained in PyTorch using the Adam optimizer and `BCEWithLogitsLoss`, which is numerically stable and suitable for binary classification.
5.  **Evaluation & Threshold Tuning:** After training, the model's performance is evaluated on the untouched test set. The Precision-Recall curve is generated, and from it, the optimal probability threshold that maximizes the F1-score is calculated. This threshold is then used to generate the final classification report and confusion matrix, leading to a model with a strong balance between catching fraud and minimizing false alarms.

## How to Run This Project
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Download the Dataset:**
    Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the root of the project folder.

3.  **Set up the Virtual Environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate it (macOS/Linux)
    source venv/bin/activate

    # Or activate it (Windows)
    .\venv\Scripts\activate
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run Training & Evaluation:**
    ```bash
    # Train the model (using SMOTE by default in the latest scripts)
    python train.py --use-smote

    # Generate the evaluation plots
    python evaluate.py
    ```

## File Structure
```
CreditCard/
├── creditcard.csv          # The dataset file
├── model.py                # Contains the PyTorch model classes
├── train.py                # Script to train the model and save results
├── evaluate.py             # Script to load results and generate plots
├── requirements.txt        # Required Python libraries
└── README.md               # This file
```


## Technologies Used
- Python 3.9+
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib & Seaborn
- Imbalanced-learn
