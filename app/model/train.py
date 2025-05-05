import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

def train_model():
    # Create sample data if the CSV doesn't exist
    data_path = 'data/housing_data.csv'
    if not os.path.exists(data_path):
        # Create directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Features that affect house prices
        sqft = np.random.randint(500, 5000, n_samples)
        bedrooms = np.random.randint(1, 7, n_samples)
        bathrooms = np.random.randint(1, 5, n_samples)
        age = np.random.randint(0, 100, n_samples)
        
        # Generate price with some noise
        price = (
            100 * sqft + 
            15000 * bedrooms + 
            20000 * bathrooms - 
            1000 * age + 
            np.random.normal(0, 50000, n_samples)
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'sqft': sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age': age,
            'price': price
        })
        
        # Save to CSV
        df.to_csv(data_path, index=False)
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model trained successfully!")
    print(f"Training R² score: {train_score:.4f}")
    print(f"Testing R² score: {test_score:.4f}")
    
    # Save the trained model
    with open('app/model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

if __name__ == "__main__":
    train_model()