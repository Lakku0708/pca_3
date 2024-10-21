# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# 2. Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. Standardize the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Accuracy, precision, and recall
    metrics = {
        'Accuracy (Train)': accuracy_score(y_train, y_pred_train),
        'Accuracy (Test)': accuracy_score(y_test, y_pred_test),
        'Precision (Test)': precision_score(y_test, y_pred_test, average='weighted'),
        'Recall (Test)': recall_score(y_test, y_pred_test, average='weighted')
    }
    return metrics

# 3. Baseline Model (No PCA)
print("Training baseline model (No PCA)...")
baseline_model = LogisticRegression(max_iter=10000)
baseline_metrics = evaluate_model(baseline_model, X_train_scaled, X_test_scaled, y_train, y_test)

print("\nBaseline Model Performance (No PCA):")
for key, value in baseline_metrics.items():
    print(f"{key}: {value:.4f}")

# 4. Apply PCA and evaluate model with different number of components
components_to_test = [2, 5, 10]
pca_metrics = []

for n_components in components_to_test:
    print(f"\nTraining model with {n_components} PCA components...")
    pca = PCA(n_components=n_components)

    # Fit PCA on training data and apply the same transformation to test data
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Train and evaluate the model on PCA-transformed data
    pca_model = LogisticRegression(max_iter=10000)
    pca_metrics.append(evaluate_model(pca_model, X_train_pca, X_test_pca, y_train, y_test))

# 5. Compare the performance of the models with different PCA components
print("\nModel Performance with PCA:")
for i, n_components in enumerate(components_to_test):
    print(f"\nPCA with {n_components} Components:")
    for key, value in pca_metrics[i].items():
        print(f"{key}: {value:.4f}")

# 6. Visualization: Plot accuracy for different numbers of components
n_components = [0] + components_to_test  # 0 represents baseline (no PCA)
accuracies = [baseline_metrics['Accuracy (Test)']] + [metrics['Accuracy (Test)'] for metrics in pca_metrics]

plt.figure(figsize=(8, 6))
plt.plot(n_components, accuracies, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs Number of PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Test Accuracy')
plt.xticks(n_components)
plt.grid(True)
plt.show()
