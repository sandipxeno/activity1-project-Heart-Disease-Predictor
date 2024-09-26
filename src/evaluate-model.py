import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model_path, test_data_path):
    # Load the trained model
    model = joblib.load(model_path)
    
    # Load the test dataset
    test_data = pd.read_csv(test_data_path)
    
    # Assuming 'Heart Disease' is the target column and the rest are features
    X_test = test_data.drop('Heart Disease', axis=1)
    y_test = test_data['Heart Disease']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf_matrix}")

if __name__ == "__main__":
    # Define the paths to the model and test data
    model_path = 'C:/Users/adrit/activity1-project-Heart-Disease-Predictor/src/trained_model.joblib'
    test_data_path = 'C:/Users/adrit/activity1-project-Heart-Disease-Predictor/data/cleaned_heart.csv'
    
    # Call the evaluation function
    evaluate_model(model_path, test_data_path)