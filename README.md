# Python_internship_task4
# NLTK-Based Chatbot

This project implements a rule-based chatbot using **NLTK (Natural Language Toolkit)**. It classifies user input into predefined categories and gives an appropriate response using Decision Tree or Naive Bayes classifiers.

---

## 🧠 Features

- Tokenization, stemming, lemmatization
- POS (Part-of-Speech) tagging
- Feature extraction
- Decision Tree and Naive Bayes classification
- Input/output chatbot interface
- Customizable dataset (`leaves.txt`)

---

## 📁 Files

| File Name      | Purpose                           |
|----------------|-----------------------------------|
| `second.py`    | Main chatbot program              |
| `leaves.txt`   | Input training data (user must add) |
| `training_data.npy` | Saved processed training data |
| `test_data.npy`     | Saved processed testing data  |

---

## 🛠️ Requirements

Install the following Python libraries:

```bash
pip install nltk pandas numpy

```
Also, download required NLTK resources:

```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('tagsets')
```

## 📄 Format for leaves.txt
The chatbot learns from a .txt file with 3 columns:
```
[input sentence] | [category] | [reply/answer]
```
Example content:
```
hi there|greeting|Hello! How can I help you?
what is your name|identity|I am a chatbot powered by NLTK.
bye|exit|Goodbye! Take care.
```
Make sure it uses the pipe symbol (|) to separate fields.

## 🚀 Running the Bot
```
python chatbot.py
```

## 🔍 How It Works
1.Reads leaves.txt data.
2Processes sentences: tokenize, remove stopwords, lemmatize, POS tag.
3Extracts features and trains 2 classifiers:
  Decision Tree
  Naive Bayes
4.Classifies new input and responds using learned replies.

## 📌 Notes
If leaves.txt file is missing or not formatted correctly, the chatbot won't work.
The project saves and loads training data as .npy using NumPy for faster loading.

## Sample output:
It starts a chat loop. Type your messages. Example:
```
you: hi there
Bot: Hello! How can I help you?

you: what is your name
Bot: I am a chatbot powered by NLTK.

you: bye
Bot: Goodbye! Take care.

```


