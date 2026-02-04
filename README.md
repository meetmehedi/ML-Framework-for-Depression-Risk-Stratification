# ML Framework for Depression Risk Stratification

An advanced machine learning framework designed to predict and analyze depression risk factors using student mental health data. This project implements a comprehensive pipeline from data preprocessing to state-of-the-art model explainability.

## 🚀 Key Features

-   **Comprehensive Pipeline**: Full data lifecycle management including cleaning, feature engineering, and scaling.
-   **Advanced Imbalance Handling**: Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) to address class imbalance in mental health datasets.
-   **Extensive Model Suite**:
    -   **10 Base Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes, MLP Classifier, XGBoost, and LightGBM.
    -   **5 Hybrid Models**: Voting Classifier, Stacking Classifier, Bagging, AdaBoost, and Extra Trees.
-   **Robust Evaluation**: Implements **Stratified 5-Fold Cross-Validation** for reliable performance metrics (Accuracy, Precision, Recall, F1-Score, and ROC-AUC).
-   **Explainable AI (XAI)**:
    -   **SHAP Analysis**: Global and local feature importance via Shapley values.
    -   **LIME Analysis**: Local Interpretable Model-agnostic Explanations for individual prediction transparency.
    -   **Feature Importance Visualization**: Integrated analysis for all tree-based ensemble models.

## 📊 Core Insights

Through rigorous analysis, the framework consistently identifies high-impact predictors for depression risk:
1.  **Anxiety Levels**: The strongest correlated factor across almost all models.
2.  **Academic Performance (CGPA)**: High influence from various CGPA ranges.
3.  **Prior Treatment History**: Whether the student has sought professional help.
4.  **Demographics**: Age and course-specific stress factors.

## 🛠️ Requirements

-   Python 3.x
-   Pandas
-   NumPy
-   Scikit-learn
-   XGBoost
-   LightGBM
-   Imbalanced-learn (SMOTE)
-   Matplotlib / Seaborn
-   SHAP
-   LIME

## 📂 Project Structure

-   `deprassion.py`: The core implementation script (Jupyter Notebook export).
-   `deprassion.ipynb`: Interactive development and visualization notebook.
-   `lime.pdf` / `lime 2.pdf`: Exported LIME explanation results.
-   `outputframe.html`: Resultant visualizations.

## 📝 How to Use

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/meetmehedi/ML-Framework-for-Depression-Risk-Stratification.git
    ```
2.  **Install dependencies**:
    ```bash
    pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn shap lime
    ```
3.  **Run the analysis**:
    ```bash
    python deprassion.py
    ```

## 🤝 Results & Visualization

The framework provides detailed confusion matrices and performance summaries both before and after applying SMOTE. The interpretability plots (SHAP/LIME) offer deep dives into *why* specific predictions are made, fostering trust and providing actionable insights for mental health screening.

---
**Author**: [meetmehedi](https://github.com/meetmehedi)
