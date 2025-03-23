from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("model/spam_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("model/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    vectorized_msg = vectorizer.transform([message])
    prediction = model.predict(vectorized_msg)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    
    return render_template('result.html', message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
