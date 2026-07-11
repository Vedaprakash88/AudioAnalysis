import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

class KNNAudioClassifier:
    """
    A K-Nearest Neighbors classifier trained on tabular audio features.
    """
    def __init__(self, csv_path, n_neighbors=10):
        self.csv_path = csv_path
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_values = None
        self.y_values = None

    def load_and_preprocess(self):
        """Loads features from CSV, standardizes columns, and encodes target labels."""
        if not self.csv_path or not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Feature CSV not found at {self.csv_path}")

        print(f"Loading data from {self.csv_path}...")
        data = pd.read_csv(self.csv_path)

        # Separate filename, features, and label
        file_names = data['filename']
        features = data.drop(['filename', 'label'], axis=1)

        # Standardize features
        standardized_features = self.scaler.fit_transform(features)

        # Build principal DataFrame
        principal_df = pd.DataFrame(data=standardized_features, columns=features.columns)
        principal_df['file_name'] = file_names
        principal_df['label'] = self.label_encoder.fit_transform(data['label'])

        # Prepare X and y
        self.x_values = principal_df.drop(['file_name', 'label'], axis=1)
        self.y_values = principal_df['label']

        # Split train/test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_values, self.y_values, test_size=0.2, random_state=42
        )
        print("Data preprocessed and split successfully.")

    def train(self):
        """Fits the KNN classifier on the training partition."""
        if self.x_train is None:
            self.load_and_preprocess()

        print(f"Training KNN Classifier with {self.n_neighbors} neighbors...")
        self.model.fit(self.x_train, self.y_train)
        print("KNN training completed.")

    def evaluate(self, save_plot_path="knn_confusion_matrix.png"):
        """Evaluates model performance and plots confusion matrix."""
        if self.model is None:
            raise ValueError("Model is not trained yet. Call train() first.")

        y_pred = self.model.predict(self.x_test)
        accuracy = self.model.score(self.x_test, self.y_test)
        conf_mat = confusion_matrix(self.y_test, y_pred)
        
        # Get target class names from label encoder
        class_names = [str(c) for c in self.label_encoder.classes_]
        class_rep = classification_report(self.y_test, y_pred, target_names=class_names)

        print("\n" + "="*40)
        print("KNN Audio Classifier Evaluation")
        print("="*40)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{conf_mat}")
        print(f"Classification Report:\n{class_rep}")

        # Heatmap plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.title('KNN Audio Feature Confusion Matrix')
        plt.tight_layout()

        if save_plot_path:
            dir_name = os.path.dirname(save_plot_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            plt.savefig(save_plot_path)
            print(f"Confusion matrix plot saved to {save_plot_path}")
        plt.close()

        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_mat
        }
