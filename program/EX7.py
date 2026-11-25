from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train SGD Classifier
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

# Predict species
y_pred = model.predict(X_test)

# Display predictions
print("Predicted Iris Species:")
for prediction in y_pred:
    print(iris.target_names[prediction])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
