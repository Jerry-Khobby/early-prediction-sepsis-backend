from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report, 
    precision_recall_curve, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def evaluate_model(predictions, y_true, threshold=0.5, filename='val'):
    predictions = predictions.flatten()
    y_true = y_true.flatten()
    y_pred = (predictions > threshold).astype(int)

    # Core metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # === 1. Classification Report ===
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # === 2. Confusion Matrix ===
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f'plots/ConfusionMatrix_{filename}.png', dpi=250)
    plt.close()

    # === 3. Precision-Recall Curve ===
    prec, rec, _ = precision_recall_curve(y_true, predictions)
    plt.figure(figsize=(7, 5))
    plt.plot(rec, prec, color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig(f'plots/PR_Curve_{filename}.png', dpi=250)
    plt.close()

    # === 4. ROC Curve ===
    fpr, tpr, _ = roc_curve(y_true, predictions)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/ROC_{filename}.png', dpi=250)
    plt.close()

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc_score,
        "confusion_matrix": cm
    }
