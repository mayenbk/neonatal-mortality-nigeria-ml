"""
models.py
Algorithm training, cross-validation, and evaluation.
Paper: Ben-Koko & Warribo (2026), WorldS4 2026, Springer LNNS, London
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

MODELS = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42)
}


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a fitted model on the test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'AUC-ROC': round(roc_auc_score(y_test, y_prob), 4)
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, cv_folds=5):
    """Train all models and evaluate on test set."""
    results = []
    for name, model in MODELS.items():
        print(f"\nTraining: {name}")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='roc_auc')
        print(f"  CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        model.fit(X_train, y_train)
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)
        print(f"  Test AUC-ROC: {result['AUC-ROC']}")
    return pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)


def get_feature_importance(X_train, y_train, feature_names):
    """Train Random Forest and return feature importance scores."""
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    importance_df['Rank'] = importance_df.index + 1
    return importance_df, rf
