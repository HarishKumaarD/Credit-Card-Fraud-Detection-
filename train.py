# train.py
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, average_precision_score
import numpy as np
from imblearn.over_sampling import SMOTE

# MODIFIED: Import the V2 models
from model import AutoencoderV2, ClassifierV2, HybridModelV2

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_and_preprocess_data(filepath, use_smote=False):
    """Loads the dataset and performs preprocessing."""
    print("Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    
    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pos_weight = None
    if use_smote:
        # NEW: Apply SMOTE to the training data
        print("Applying SMOTE to training data...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("SMOTE applied. New training set size:", X_train.shape)
    else:
        # MODIFIED: Calculate class weight only if not using SMOTE
        print("Using class weighting strategy.")
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
        print(f"Positive class weight for loss function: {pos_weight.item():.2f}")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, pos_weight

def train_model(model, train_loader, pos_weight, epochs, lr):
    """Trains the hybrid model."""
    print("\nStarting model training...")
    # MODIFIED: Use weight only if it's provided (i.e., not using SMOTE)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.6f}")

def evaluate_and_save(model, X_test_tensor, y_test_tensor):
    """Evaluates the model and prints a classification report."""
    print("\nEvaluating model on the test set...")
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        predicted_probs = torch.sigmoid(logits).squeeze().numpy()
        predicted_labels = (predicted_probs > 0.5).astype(int)
        
    y_true = y_test_tensor.squeeze().numpy()
    
    print("\nClassification Report (at 0.5 threshold):")
    print(classification_report(y_true, predicted_labels, target_names=['Not Fraud (0)', 'Fraud (1)']))
    
    pr_auc = average_precision_score(y_true, predicted_probs)
    print(f"Average Precision (PR AUC): {pr_auc:.4f}")
    
    np.savez('results.npz', y_true=y_true, predicted_probs=predicted_probs)
    print("\nEvaluation results saved to results.npz")

def main():
    """Main function to run the entire pipeline."""
    parser = argparse.ArgumentParser(description="Imbalanced Credit Card Fraud Detection Training Script")
    parser.add_argument('--filepath', type=str, default='creditcard.csv', help='Path to the dataset CSV file.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    # NEW: Argument to choose between class weighting and SMOTE
    parser.add_argument('--use-smote', action='store_true', help='Use SMOTE for oversampling instead of class weighting.')
    args = parser.parse_args()

    set_seed(args.seed)
    
    X_train, y_train, X_test, y_test, pos_weight = load_and_preprocess_data(args.filepath, args.use_smote)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # MODIFIED: Initialize the V2 models
    input_size = X_train.shape[1]
    encoding_size = 4  # A smaller encoding size is better for this refined model
    autoencoder = AutoencoderV2(input_size, encoding_size)
    classifier = ClassifierV2(encoding_size)
    hybrid_model = HybridModelV2(autoencoder, classifier)
    
    train_model(hybrid_model, train_loader, pos_weight, args.epochs, args.lr)
    evaluate_and_save(hybrid_model, X_test, y_test)

if __name__ == '__main__':
    main()