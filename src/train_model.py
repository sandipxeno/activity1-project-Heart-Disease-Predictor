import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_and_save_model(input_data_path, model_output_path):
    # Load the dataset
    data = pd.read_csv(input_data_path)
    
    # Prepare the data (Assuming 'Heart Disease' is the target column)
    X = data.drop('Heart Disease', axis=1)
    y = data['Heart Disease']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a RandomForest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    input_data_path = 'C:/Users/mrxen/OneDrive/Desktop/activity1-project-Heart-Disease-Predictor.git/activity1-project-Heart-Disease-Predictor/data/cleaned_heart.csv'
    model_output_path = 'C:/Users/mrxen/OneDrive/Desktop/activity1-project-Heart-Disease-Predictor.git/activity1-project-Heart-Disease-Predictor/src/trained_model.joblib'
    
    train_and_save_model(input_data_path, model_output_path)