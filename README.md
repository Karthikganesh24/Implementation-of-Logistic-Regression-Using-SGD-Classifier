# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as NumPy, Scikit-learn, and the Iris dataset.
2. Load the Iris dataset and split it into training and testing datasets.
3. Create and train the SGD Classifier model using the training dataset.
4. Predict the Iris flower species using the test data and display the predicted results.

## Program:
```
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

```

## Output:
Implementation-of-Logistic-Regression-Using-SGD-Classifier/output/Screenshot 2026-11-25 100200.png

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.