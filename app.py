from flask import Flask, request, render_template
import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained model
model = load_model('learned_model.h5')

# Load the tokenizer
tokenizer = Tokenizer()
with open('Pride & Predjuice.txt', 'r', encoding='ISO-8859-1') as f:
    data = f.read()
tokenizer.fit_on_texts([data])

# Set the maximum sequence length
max_seq_len = 2

# Set the vocabulary size
vocab_size = len(tokenizer.word_index) + 1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    n_pred = 1  # Generate one word prediction
    in_text = input_text
    # generate a fixed number of words
    for _ in range(n_pred):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_seq_len, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == np.argmax(yhat):
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
    
    return render_template('index.html', prediction=in_text)

if __name__ == '__main__':
    app.run(debug=True)
