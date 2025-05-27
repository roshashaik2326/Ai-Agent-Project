# IMDB Sentiment Analyzer: Extended Documentation

---

## 1. Project Content

The IMDB Sentiment Analyzer project applies deep learning to natural language processing (NLP) for movie review sentiment classification. It predicts whether a given movie review from the IMDB dataset is positive or negative and offers insight into the language patterns influencing these predictions.

**Included Components:**
- Data loading, preprocessing, and tokenization scripts
- LSTM model definition and training scripts
- Evaluation and prediction modules
- Streamlit web app for interactive sentiment analysis
- Experiment notebooks and visualizations

---

## 2. Project Code

The codebase is organized for clarity and experimentation:

- **data_processing.py**  
  Handles loading the IMDB dataset, cleaning reviews, tokenizing text, and padding sequences for model input.

  _Example:_  
  ```python
  from tensorflow.keras.preprocessing.text import Tokenizer
  from tensorflow.keras.preprocessing.sequence import pad_sequences

  # Suppose reviews is a list of text reviews
  tokenizer = Tokenizer(num_words=10000)
  tokenizer.fit_on_texts(reviews)
  sequences = tokenizer.texts_to_sequences(reviews)
  padded = pad_sequences(sequences, maxlen=200)
  ```
  > This snippet tokenizes text data and pads sequences to ensure equal length input for the neural network.

- **lstm_model.py**  
  Defines the LSTM (Long Short-Term Memory) neural network for sequence prediction.

  _Example:_  
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Embedding, LSTM, Dense

  model = Sequential([
      Embedding(input_dim=10000, output_dim=128, input_length=200),
      LSTM(64),
      Dense(1, activation='sigmoid')
  ])
  ```
  > Embedding layer converts words to vectors; LSTM learns temporal dependencies; Dense outputs sentiment.

- **train.py**  
  Trains the LSTM model, tracks accuracy and loss, and saves the best model for inference.

  _Example:_  
  ```python
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  model.fit(padded, labels, epochs=5, batch_size=64, validation_split=0.2)
  ```
  > Model is trained for sentiment classification on padded review data.

- **analyze.py**  
  Loads the trained model and predicts sentiment for new reviews, displaying results and confidence scores.

  _Example:_  
  ```python
  import numpy as np
  new_review = ["A masterpiece of filmmaking!"]
  seq = tokenizer.texts_to_sequences(new_review)
  pad = pad_sequences(seq, maxlen=200)
  prediction = model.predict(pad)
  print("Positive" if prediction[0][0] > 0.5 else "Negative")
  ```
  > Predicts whether the new review is positive or negative.

- **app.py**  
  Streamlit web app for user-friendly review input, instant sentiment prediction, and visualization.

---

## 3. Key Technologies

- **Python 3.x:** Core scripting language.
- **TensorFlow/Keras:** For building, training, and evaluating LSTM models.
- **Tokenization & Padding:** For text preprocessing.
- **NumPy & Pandas:** For data handling and manipulation.
- **Streamlit:** For building an interactive web app.
- **Matplotlib/Seaborn:** For visualizing training metrics.

---

## 4. Description

This project demonstrates how deep learning can be applied to NLP tasks like sentiment analysis. The workflow is as follows:

1. **Data Preparation:**  
   - Loads IMDB movie reviews.
   - Cleans text, tokenizes words, and pads sequences for uniform input.

2. **Model Training:**  
   - Uses an Embedding layer to convert words into vector representations.
   - LSTM layer captures sequential dependencies in reviews.
   - Dense layer outputs a probability (positive/negative).

3. **Prediction:**  
   - New reviews are preprocessed and run through the model for instant sentiment classification.
   - Output includes sentiment label and confidence score.

4. **Deployment:**  
   - Streamlit app allows users to enter reviews and visualize predictions.

---

## 5. Output

### Model Performance

- **Accuracy:** Achieves ~87% on test data.
- **Training Metrics:** Accuracy and loss curves are plotted to monitor learning.

### Example Predictions

- **Sample Input:**  
  `"The movie was thrilling and well-acted!"`  
  **Prediction:** Positive (0.95)

- **Sample Input:**  
  `"A dull and uninspired sequel."`  
  **Prediction:** Negative (0.10)

### Streamlit App

- Users can type or paste a review and receive an instant sentiment prediction.

---

## 6. Further Research

- **Transformer Models:** Experiment with BERT, RoBERTa, or similar for improved accuracy.
- **Aspect-Based Sentiment:** Extract sentiment toward specific aspects (e.g., plot, acting).
- **Multilingual Support:** Extend the approach to non-English reviews.
- **Explainability:** Use attention weights or LIME to highlight influential words/phrases.
- **Dataset Expansion:** Incorporate other review sources for broader generalization.

---

## 7. Sample Code Snippets and Explanations

### Tokenization and Padding

```python
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
padded = pad_sequences(sequences, maxlen=200)
```
*Converts text reviews into integer sequences and pads them for the model.*

### Model Definition

```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
```
*A simple LSTM model for binary sentiment classification.*

### Making a Prediction

```python
new_review = ["What a fantastic movie!"]
seq = tokenizer.texts_to_sequences(new_review)
pad = pad_sequences(seq, maxlen=200)
prediction = model.predict(pad)
print("Positive" if prediction[0][0] > 0.5 else "Negative")
```
*Predicts sentiment for new, unseen reviews.*

---

## 8. Deployment & Usage

**Run the Streamlit App:**
1. Clone the repository:
   ```
   git clone https://github.com/roshashaik2326/Ai-Agent-Project.git
   ```
2. Install dependencies:
   ```
   cd imdb_sentiment
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

**Colab Notebook:**
- Open the provided Colab link.
- Run all cells for full training, evaluation, and interactive testing.

---

## 9. References

- [TensorFlow Keras Documentation](https://keras.io/)
- [IMDB Dataset Info](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Tokenization Guide](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)

---

## 10. Contact & Contributions

- **Author:** [roshashaik2326](https://github.com/roshashaik2326)
- **Feedback/Issues:** Use [GitHub Issues](https://github.com/roshashaik2326/Ai-Agent-Project/issues)
- **Contributions:** Fork, improve, and submit a pull request!

---

> For questions or suggestions about the IMDB Sentiment Analyzer, contact via GitHub. Explore, experiment, and enhance NLP with deep learning!
