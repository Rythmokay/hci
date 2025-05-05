# House Price Prediction App

A minimalist web application for predicting house prices using Linear Regression.

## Project Structure

```
hci/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── model.pkl 
│   ├── static/
│   │   └── style.css
│   └── templates/
│       └── index.html
├── data/
│   └── housing_data.csv 
├── requirements.txt
└── README.md
```

## Features

- Predicts house prices based on square footage, number of bedrooms, number of bathrooms, and house age
- Simple, clean user interface
- FastAPI backend with automatic API documentation
- Linear Regression model trained on sample data

## Installation

1. Clone this repository:
```
git clone https://github.com/Rythmokay/hci.git
cd hci
```

2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

## Running the Application

Run the application with the following command:

```
python -m app.main
```

The application will be available at http://localhost:8000

When started for the first time, the application will:
1. Generate sample housing data (if none exists)
2. Train a Linear Regression model on this data
3. Save the trained model for future use

## API Documentation

FastAPI automatically generates API documentation for the application. You can access it at:

- http://localhost:8000/docs - Swagger UI
- http://localhost:8000/redoc - ReDoc UI

## License

MIT
