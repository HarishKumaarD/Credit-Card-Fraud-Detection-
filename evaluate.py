# evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score, classification_report

def find_optimal_threshold(y_true, y_probs):
    """Finds the optimal probability threshold to maximize F1-score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    # Calculate F1 score for each threshold, adding a small epsilon to avoid division by zero
    f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
    # The last threshold is 1.0, which gives recall 0. We exclude it.
    best_f1_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    return best_threshold, best_f1

def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    """Plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

def plot_roc_curve(y_true, y_probs):
    """Plots the ROC curve."""
    roc_auc = roc_auc_score(y_true, y_probs)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_precision_recall_curve(y_true, y_probs, optimal_threshold):
    """Plots the Precision-Recall curve and the optimal threshold point."""
    pr_auc = average_precision_score(y_true, y_probs)
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Find the precision and recall for the optimal threshold
    # Find the index of the threshold closest to our optimal_threshold
    idx = np.argmin(np.abs(thresholds - optimal_threshold))
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    # Plot the optimal point
    plt.scatter(recall[idx], precision[idx], marker='o', color='red', label=f'Optimal Threshold ({optimal_threshold:.2f})', zorder=5)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """Load results and generate plots."""
    try:
        data = np.load('results.npz')
    except FileNotFoundError:
        print("Error: results.npz not found. Please run train.py first to generate the results.")
        return
        
    y_true = data['y_true']
    predicted_probs = data['predicted_probs']
    class_names = ['Not Fraud', 'Fraud']
    
    print("--- Performance Analysis ---")
    
    # --- Evaluation at default 0.5 threshold ---
    print("\n--- Results at Default Threshold (0.5) ---")
    default_labels = (predicted_probs > 0.5).astype(int)
    print(classification_report(y_true, default_labels, target_names=class_names))
    
    # --- Find and evaluate at optimal threshold ---
    # NEW: Find optimal threshold based on F1-score
    optimal_thresh, max_f1 = find_optimal_threshold(y_true, predicted_probs)
    print(f"\nOptimal threshold to maximize F1-score found at: {optimal_thresh:.4f} (Max F1-Score: {max_f1:.4f})")
    
    print(f"\n--- Results at Optimal Threshold ({optimal_thresh:.2f}) ---")
    optimal_labels = (predicted_probs > optimal_thresh).astype(int)
    print(classification_report(y_true, optimal_labels, target_names=class_names))
    
    # --- Generate Visualizations ---
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_true, optimal_labels, class_names, title='Confusion Matrix (at Optimal Threshold)')
    plot_roc_curve(y_true, predicted_probs)
    plot_precision_recall_curve(y_true, predicted_probs, optimal_thresh)
    print("Done.")

if __name__ == '__main__':
    main()