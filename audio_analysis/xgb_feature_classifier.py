import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier, plot_importance

class XGBFeatureClassifier:
    """
    Trains, evaluates, and plots feature importance for an XGBoost Classifier on tabular audio features.
    """
    def __init__(self, csv_path, model_params=None):
        self.csv_path = csv_path
        self.model_params = model_params if model_params is not None else {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'verbosity': 1,
            'booster': 'gbtree',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'alpha': 0,
            'objective': 'multi:softmax',
            'eval_metric': 'logloss',
            'nthread': 16
        }
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Split data storage
        self.x_values = None
        self.y_values = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.file_names = None

    def load_and_preprocess(self):
        """Loads features from CSV, standardizes columns, and encodes target labels."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Feature CSV not found at {self.csv_path}")

        print(f"Loading data from {self.csv_path}...")
        data = pd.read_csv(self.csv_path)

        # Separate filename, features, and label
        self.file_names = data['filename']
        # Also drop PCA columns if they are present in the CSV
        cols_to_drop = ['filename', 'label']
        for pca_col in ['principal component 1', 'principal component 2']:
            if pca_col in data.columns:
                cols_to_drop.append(pca_col)
                
        features = data.drop(cols_to_drop, axis=1)

        # Standardize features
        standardized_features = self.scaler.fit_transform(features)

        # Build principal DataFrame
        principal_df = pd.DataFrame(data=standardized_features, columns=features.columns)
        principal_df['file_name'] = self.file_names
        principal_df['label'] = self.label_encoder.fit_transform(data['label'])

        # Prepare X and y
        self.x_values = principal_df.drop(['file_name', 'label'], axis=1)
        self.y_values = principal_df['label']

        # Split train/test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_values, self.y_values, test_size=0.2, random_state=42
        )
        print("Data preprocessed and split successfully.")

    def train(self, verbose=True):
        """Fits the XGBoost Classifier on the training partition."""
        if self.x_train is None:
            self.load_and_preprocess()

        print("Training XGBoost Classifier...")
        params = self.model_params.copy()
        unique_classes = len(np.unique(self.y_values))
        if 'num_class' not in params and params.get('objective') in ['multi:softmax', 'multi:softprob']:
            params['num_class'] = unique_classes

        self.model = XGBClassifier(**params)
        self.model.fit(self.x_train, self.y_train, verbose=verbose)
        print("Training completed.")

    def evaluate(self, run_cv=True):
        """Evaluates model performance and prints reports."""
        if self.model is None:
            self.train()

        y_pred = self.model.predict(self.x_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        conf_mat = confusion_matrix(self.y_test, y_pred)
        class_rep = classification_report(self.y_test, y_pred)
        accuracy_via_score = self.model.score(self.x_test, self.y_test)

        print("\n" + "="*40)
        print("XGBoost Evaluation Results")
        print("="*40)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Accuracy via Score: {accuracy_via_score:.4f}")
        print(f"Confusion Matrix:\n{conf_mat}")
        print(f"Classification Report:\n{class_rep}")

        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_mat,
            'classification_report': class_rep,
        }

        if run_cv:
            print("Running 10-fold cross-validation...")
            cv_scores = cross_val_score(self.model, self.x_values, self.y_values, cv=10)
            print(f"Cross-validated accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            results['cv_accuracy_mean'] = cv_scores.mean()
            results['cv_accuracy_std'] = cv_scores.std()

        return results

    def plot_feature_importance(self, output_path="important_features.png", max_num_features=10):
        """Plots and saves feature importance chart."""
        if self.model is None:
            raise ValueError("Model is not trained yet. Call train() first.")

        fig, ax = plt.subplots(figsize=(10, 8))
        plot_importance(self.model, ax=ax, max_num_features=max_num_features)
        plt.title("XGBoost Feature Importance")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Feature importance plot saved to {output_path}")
