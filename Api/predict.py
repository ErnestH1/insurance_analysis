import requests

# Define input data
input_data = {
    "age": 35,
    "sex": "male",
    "bmi": 25.5,
    "children": 2,
    "smoker": "no",
    "region": "southwest"
}

# Make POST request to FastAPI server for predictions
response = requests.post("http://127.0.0.1:8000/predict/", json=input_data)

# Print predictions
print(response.json())
