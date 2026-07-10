from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, classification_report,
)


def evaluate(y_true, y_prob, threshold: float = 0.5) -> float:
    """Print fraud-relevant metrics. Returns PR-AUC (the north-star metric)."""
    y_pred = (y_prob >= threshold).astype(int)
    print(f"Threshold: {threshold}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_true, y_prob):.4f}")
    print("\nConfusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_true, y_pred))
    print("\n", classification_report(y_true, y_pred, digits=4))
    return average_precision_score(y_true, y_prob)