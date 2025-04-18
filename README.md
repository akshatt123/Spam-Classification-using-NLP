# 📩 Spam Classification using NLP

🚀 A machine learning project that classifies messages as **Spam or Ham** using **Natural Language Processing (NLP)**.\
This project implements **Bag of Words, TF-IDF, and Word2Vec** along with **Naïve Bayes & Random Forest** classifiers. The best accuracy was achieved using **Random Forest with TF-IDF**.

It also includes a **Flask web app** where users can input a message and get instant spam predictions. The UI is **modern, responsive, and interactive**. 🎨✨

---

## ⚡ Features

✔️ **Spam Detection** using advanced NLP techniques\
✔️ Implements **TF-IDF, Bag-of-Words & Word2Vec**\
✔️ Uses **Naïve Bayes & Random Forest** classifiers\
✔️ **Flask Web App** for real-time predictions\
✔️ **Modern & Interactive UI** with smooth animations\
✔️ Saves the trained **model as ********************`.pkl`******************** file** for future use

---

## 🛠️ Tech Stack

- **Python** (scikit-learn, pandas, NumPy)
- **Machine Learning** (Random Forest, Naïve Bayes)
- **NLP** (TF-IDF, Word2Vec, Bag-of-Words)
- **Flask** (Web App Backend)
- **HTML, CSS, JavaScript** (Frontend)

---

## 💁️‍♂️ Project Structure

```
Spam-Classification/
│── model/
│   ├── spam_classifier.pkl   # Saved trained model
│   ├── vectorizer.pkl        # Saved TF-IDF vectorizer
│── templates/
│   ├── index.html            # Web UI for spam prediction
│   ├── result.html           # Displays prediction results
│── static/
│   ├── style.css             # Custom CSS styles
│── app.py                    # Flask app for API & UI
│── train_model.py            # Model training script
│── README.md                 # Project Documentation
│── requirements.txt          # Dependencies
```

---

## 🚀 Installation & Setup

1️⃣ **Clone the repository**

```sh
git clone https://github.com/akshatt123/spam-classification-nlp.git
cd spam-classification-nlp
```

2️⃣ **Install Dependencies**

```sh
pip install -r requirements.txt
```

3️⃣ **Train the Model**

```sh
python train_model.py
```

This will save the **spam\_classifier.pkl** model and **vectorizer.pkl**.

4️⃣ **Run the Flask App**

```sh
python app.py
```


---

## 🌟 Usage

1. Open the web app in your browser.
2. Enter a message in the text box.
3. Click **"Check Spam"**.
4. See if the message is **Spam or Not Spam**.

---


## 📌 Future Enhancements

💡 Deploy the model using **AWS/GCP**\
💡 Add **LSTM-based Deep Learning Model**\
💡 Improve UI with **React or Vue.js**

