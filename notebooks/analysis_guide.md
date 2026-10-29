# Analysis Guide

This notebook walks through the full ML pipeline for neonatal mortality risk prediction.

## Steps

1. Load and inspect the NDHS data
2. Apply feature engineering and preprocessing
3. Handle class imbalance with SMOTE
4. Train and evaluate five classification algorithms
5. Analyse feature importance from Random Forest
6. Visualise results

## Running the Pipeline

```python
from src.preprocessing import prepare_features, split_and_resample
from src.models import train_and_evaluate, get_feature_importance
from src.visualisation import plot_model_comparison, plot_feature_importance

# Load and prepare data
X, y = prepare_features('data/ndhs_indicators.csv')
X_train, X_test, y_train, y_test = split_and_resample(X, y)

# Train and evaluate
results = train_and_evaluate(X_train, X_test, y_train, y_test)
print(results)

# Feature importance
importance_df, rf_model = get_feature_importance(X_train, y_train, X.columns.tolist())
plot_feature_importance(importance_df)
```
