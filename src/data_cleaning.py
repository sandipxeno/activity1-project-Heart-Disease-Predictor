import pandas as pd

def load_and_clean_data(input_path, output_path):
    # Load the dataset
    data = pd.read_csv(input_path)
    
    # Drop rows with missing values
    data_cleaned = data.dropna()

    # Handle categorical variables
    data_cleaned = pd.get_dummies(data_cleaned, drop_first=True)

    # Save the cleaned dataset
    data_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Use either a relative or absolute path
    input_path = 'C:/Users/mrxen/OneDrive/Desktop/activity1-project-Heart-Disease-Predictor.git/activity1-project-Heart-Disease-Predictor/data/heart_disease_dataset.csv'
    output_path = 'C:/Users/mrxen/OneDrive/Desktop/activity1-project-Heart-Disease-Predictor.git/activity1-project-Heart-Disease-Predictor/data/cleaned_heart.csv'
    
    # Call the function to clean and save the dataset
    load_and_clean_data(input_path, output_path)

