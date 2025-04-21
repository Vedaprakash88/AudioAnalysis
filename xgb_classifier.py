from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt


# Load your CSV into a DataFrame
data = pd.read_csv('D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\Data\\PCA_features_30_sec.csv')

# Extract features (excluding filename and label)
features = data.drop(['filename', 'label', 'principal component 1', 'principal component 2'], axis=1)

# Standardize features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# Apply PCA
#pca = PCA(n_components=5)#
#principal_components = pca.fit_transform(standardized_features)

# Add principal components back to DataFrame
principal_df = pd.DataFrame(data=standardized_features)
principal_df['file_name'] = data['filename']
principal_df['label'] = LabelEncoder().fit_transform(data ['label'])

principal_df.to_csv('D:\\10. SRH_Academia\\1. All_Notes\\2. Semester 2\\3. Artificial Intelligence\\Project\\DATA\\Data\\PCA_features_30_sec_new.csv')

x_values = principal_df.drop(['file_name', 'label'], axis=1)
y_values = principal_df['label']

x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=42)

model = XGBClassifier(
    max_depth=6,                # Maximum depth of a tree
    learning_rate=0.1,          # Step size shrinkage
    n_estimators=500,           # Number of boosting rounds
    verbosity=1,                # Print information during training
    booster='gbtree',           # Type of booster ('gbtree', 'gblinear', 'dart')
    subsample=0.8,              # Fraction of samples used for training each tree
    colsample_bytree=0.8,       # Fraction of features used for training each tree
    gamma=0,                    # Minimum loss reduction for a split
    # lambda=1,                   # L2 regularization term on weights
    alpha=0,                    # L1 regularization term on weights
    objective='multi:softmax',  # Loss function for multi classification
    eval_metric='logloss',      # Evaluation metric
    nthread=16 
    )
model.fit(
    x_train, 
    y_train,
    verbose=True)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
class_rep = classification_report(y_test, y_pred)
cv_score = cross_val_score(model, x_values, y_values, cv=10)
accuracy_via_score = model.score(x_test, y_test)

print(f"Accuracy: {accuracy:.2f}")
print(f'confusion_matrix:\n {conf_mat}')
print(f'classification_report: \n {class_rep}')
print(f"Cross-validated accuracy: {cv_score.mean():.2f} ± {cv_score.std():.2f}")
print(f"Accuracy via score method: {accuracy_via_score:.2f}")
plot_importance(model, max_num_features=10)
plt.show()
