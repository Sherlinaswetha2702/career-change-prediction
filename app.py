from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load the trained model
model_path = "naive_bayes_model1.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load the dataset to get label encoders for categorical variables
df = pd.read_csv("career_change_prediction_dataset.csv")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# Fit LabelEncoders to match training transformation
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# StandardScaler for numerical features
numerical_columns = [
    'Age', 'Years of Experience', 'Job Satisfaction', 'Work-Life Balance',
    'Job Opportunities', 'Salary', 'Job Security', 'Career Change Interest',
    'Skills Gap', 'Mentorship Available', 'Certifications',
    'Freelancing Experience', 'Geographic Mobility', 'Professional Networks',
    'Career Change Events', 'Technology Adoption'
]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        form_data = request.form.to_dict()
        
        # Convert categorical values using LabelEncoders
        for col in categorical_cols:
            if col in form_data:
                form_data[col] = label_encoders[col].transform([form_data[col]])[0]
        
        # Convert numerical values
        for col in numerical_columns:
            form_data[col] = float(form_data[col])
        
        # Convert to DataFrame
        input_data = pd.DataFrame([form_data])
        
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction_text = "Likely to Change Occupation" if prediction[0] == 1 else "Unlikely to Change Occupation"
        
        return render_template('index.html', prediction_text=f'Prediction: {prediction_text}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":

    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)

