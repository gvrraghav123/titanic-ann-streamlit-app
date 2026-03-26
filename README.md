# Titanic Survival Prediction (ANN)

This project predicts whether a passenger survived the Titanic disaster using an Artificial Neural Network (ANN).

## Tech Stack
- Python
- TensorFlow / Keras
- Scikit-learn
- Streamlit

## Features Used
- Passenger Class (Pclass)
- Sex
- Siblings/Spouses (SibSp)
- Parents/Children (Parch)
- Fare
- Embarked

## Model
- ANN with multiple hidden layers
- Activation: ReLU (hidden), Sigmoid (output)
- Loss: Binary Crossentropy
- Optimizer: Adam

## Run Locally

pip install -r requirements.txt  
streamlit run app.py

## Deployment
Deploy easily using Streamlit Cloud:
1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Select repo
4. Deploy

## Files
- app.py → Streamlit app
- model.keras → Trained ANN model
- encoders.pkl → Preprocessing objects
- scaler.pkl → Feature scaler

## Output
- Predicts survival probability
- Displays confidence score
