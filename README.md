# Neonatal Mortality in Nigeria: A Machine Learning Framework

A machine learning framework for neonatal mortality risk prediction using indicators from the 2023–24 Nigeria Demographic and Health Survey (NDHS).

**Published in:** WorldS4 2026, Springer LNNS | *Neonatal Mortality in Nigeria: A Machine Learning Framework for Risk Prediction Using National Survey Data* | Accepted & Presented, London, UK, July 2026.

## Overview

This repository contains the full machine learning pipeline developed for the paper, including:
- Feature selection and preprocessing
- SMOTE-based class imbalance handling
- Comparative evaluation of 5 classification algorithms
- Feature importance analysis using Random Forest

## Results Summary

| Algorithm | Accuracy | AUC-ROC | F1-Score |
|-----------|----------|---------|---------|
| Logistic Regression | 76.4% | 0.78 | 0.69 |
| Decision Tree | 74.1% | 0.75 | 0.70 |
| **Random Forest** | **83.7%** | **0.89** | **0.80** |
| Gradient Boosting | 81.2% | 0.86 | 0.78 |
| Support Vector Machine | 77.9% | 0.80 | 0.72 |

**Best model:** Random Forest (AUC-ROC: 0.89)

**Top predictors:**
1. Skilled Birth Attendance (0.187)
2. Preceding Birth Interval (0.163)
3. Antenatal Care Visits 4+ (0.148)

## Dataset

This framework uses indicators from the [2023–24 Nigeria Demographic and Health Survey (NDHS)](https://microdata.worldbank.org/index.php/catalog/8029), published by the National Population Commission (NPC) and ICF.

Full microdata is available via the [DHS Programme data request process](https://dhsprogram.com/data/new-user-registration.cfm).

## Repository Structure

```
neonatal-mortality-nigeria-ml/
├── README.md
├── CITATION.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   └── visualisation.py
├── data/
│   └── README.md
├── notebooks/
│   └── analysis_guide.md
└── results/
    └── model_comparison.csv
```

## Setup

```bash
git clone https://github.com/mayenbk/neonatal-mortality-nigeria-ml
cd neonatal-mortality-nigeria-ml
pip install -r requirements.txt
```

## Citation

See `CITATION.md`

## Author

**Mayen Ben-Koko** 
[linkedin.com/in/mayenbk](https://linkedin.com/in/mayenbk) | [github.com/mayenbk](https://github.com/mayenbk)

## License

MIT License
