import pandas as pd
load = pd.read_csv('BrainTumor.csv')
# Display the first few rows of the dataset
print(load.head())
# Display the shape of the dataset
print("Shape of the dataset:", load.shape)
# Display the columns of the dataset
print("Columns in the dataset:", load.columns.tolist())
# Display the data types of each column
print("Data types of each column:\n", load.dtypes)
# Display basic statistics of the dataset
print("Basic statistics of the dataset:\n", load.describe())
# Display the number of missing values in each column
print("Number of missing values in each column:\n", load.isnull().sum())

#preprocessing
load.drop('Image',axis=1, inplace=True)    
from sklearn.model_selection import train_test_split
X = load.drop('Class', axis=1)
y = load['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Display the shapes of the training and testing sets
print("Shape of training set:", X_train.shape, y_train.shape)
# Display the first few rows of the training set
print("First few rows of the training set:\n", X_train.head())
#Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# Encoding
for column in load.select_dtypes(include=['object']).columns:
    load[column] = label_encoder.fit_transform(load[column])
# Display the first few rows of the dataset after encoding
print("First few rows of the dataset after encoding:\n", load.head())

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Display the first few rows of the scaled training set
print("First few rows of the scaled training set:\n", pd.DataFrame(X_train_scaled, columns=X_train.columns).head())

#Training a machine learning model
#Neural Network
from sklearn.neural_network import MLPClassifier
# Create a neural network model
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
# Train the model
nn_model.fit(X_train_scaled, y_train)
# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Make predictions on the test set
y_pred = nn_model.predict(X_test_scaled)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the neural network model:", accuracy)
# Display classification report
print("Classification report:\n", classification_report(y_test, y_pred))
# Display confusion matrix
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))






