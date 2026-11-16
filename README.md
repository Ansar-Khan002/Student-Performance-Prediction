# Student Performance Prediction System

A beginner-level Machine Learning project that predicts a student's final grade (G3) from the UCI Student Performance dataset.

## Dataset
`data/student_combined.csv` - 1,044 students (395 maths + 649 portuguese), 33 features.
Target variable: G3 (final grade, 0-20).

## Project Structure
```
main.py       -> trains the models and runs the prediction function
second.py     -> creates all charts and saves them in visualizations/
data/         -> dataset
visualizations/ -> saved charts (PNG)
```

## How to Run
```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python second.py    # creates the charts
python main.py      # trains models and prints predictions
```

## Workflow
Data -> Cleaning -> EDA -> Feature Encoding -> Train/Test Split -> Model Training -> Evaluation -> Prediction

## Model Results (80/20 split, random_state=42)
| Model | R2 | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.800 | 1.017 | 1.760 |
| Decision Tree | 0.709 | 1.134 | 2.120 |
| Random Forest | 0.827 | 0.937 | 1.636 |

Random Forest gives the best result. Train R2 is 0.980 against test R2 0.827, so the model fits the training data a bit too closely. This is expected with 100 trees on a small dataset, but test accuracy still holds up.

## Prediction System
`predict_grade()` takes one student's details and returns a grade out of 20 with a Pass / At risk label (below 10 is At risk).

Sample output:
```
Good student: 14.8 / 20  ->  Pass
Weak student: 5.9 / 20  ->  At risk
```

## Charts
1. Final grade distribution
2. Study time vs final grade
3. Past failures vs final grade
4. Second term grade vs final grade
5. Maths vs Portuguese
6. Correlation heatmap

## What I Learned
G1 and G2 are by far the strongest predictors of G3 (0.91 and 0.81 correlation). Past failures have a clear negative effect, study time helps up to level 3, and absences alone have almost no effect (-0.05).
