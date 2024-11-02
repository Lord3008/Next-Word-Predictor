## Next Word Predictor:
Author: Lord Sen
<br><br><br>
This is an NLP based project done by me on a small scale to predict the next words we are going to type.

A Next Word Predictor project using Natural Language Processing (NLP) is a common application that aims to predict the next word in a sentence given the preceding words. This project involves various NLP techniques and models to analyze and generate text. Here’s a short note on the key aspects of such a project:

### Key Components:

1. **Data Collection**:
   - **Corpus Selection**: Choose a large and diverse text corpus (e.g., books, articles, or scraped web content) to train the model.
   - **Preprocessing**: Clean the text data by removing punctuation, converting to lowercase, tokenizing sentences and words, and handling special characters.

2. **Text Preprocessing**:
   - **Tokenization**: Split text into individual words or tokens.
   - **Stopword Removal**: Remove common words (e.g., "and", "the") that do not carry significant meaning.
   - **Stemming and Lemmatization**: Reduce words to their root form to ensure consistency.

3. **Model Selection**:
   - **N-gram Models**: Use statistical models like bigrams or trigrams, which predict the next word based on the previous one or two words.
   - **Neural Networks**: Employ models such as Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), or Transformers to capture more complex patterns in the text.

4. **Training the Model**:
   - **Feature Extraction**: Convert text into numerical representations (e.g., one-hot encoding, word embeddings).
   - **Training**: Train the model on the preprocessed text data to learn word patterns and context.
   - **Validation**: Use a validation set to tune hyperparameters and prevent overfitting.

5. **Evaluation**:
   - **Perplexity**: Measure how well the model predicts a sample, with lower values indicating better performance.
   - **Accuracy**: Evaluate the model's ability to predict the correct next word.

6. **Deployment**:
   - **Interactive Interface**: Create a user-friendly interface where users can input text and get next word predictions.
   - **API**: Develop an API to integrate the predictor into other applications.

### Example Implementation:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Sample corpus
corpus = [
    "I love machine learning",
    "I love natural language processing",
    "Machine learning is fascinating",
    "Natural language processing is a subset of machine learning"
]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences and create predictors and labels
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Model building
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
model.fit(X, y, epochs=200, verbose=1)

# Predicting next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word = tokenizer.index_word[np.argmax(predicted)]
    return predicted_word

# Example usage
input_text = "I love"
next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
print(f"Next word prediction for '{input_text}': {next_word}")
```

### Conclusion:
A Next Word Predictor using NLP leverages techniques from text processing and machine learning to build models that can predict the next word in a sentence. By preprocessing text data, training models, and evaluating their performance, developers can create applications that enhance user experience in various contexts, such as text editors, chatbots, and language learning tools.

This project is undergoing constant improvement to make it more efficient and create something new using this project to solve some problems we face while typing.
