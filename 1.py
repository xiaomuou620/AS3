import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
def load_dataset(filename):
    samples, labels = [], []
    with open(filename, 'r') as file:
        for line in file:
            sample, label = line.strip().split(',')
            samples.append(sample)
            labels.append(int(label))
    return samples, labels

# Preprocess the dataset
def preprocess_data(samples, labels, max_length=20):
    char_to_int = {'a': 1, 'b': 2, 'c': 3}
    X = [[char_to_int[char] for char in sample] for sample in samples]
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_length, padding='post')
    y = np.array(labels)
    return X, y

# Define the Vanilla RNN model
def create_rnn_model(input_length):
    model = Sequential([
        Embedding(input_dim=4, output_dim=4, input_length=input_length),
        SimpleRNN(50, activation='tanh'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define the LSTM model
def create_lstm_model(input_length):
    model = Sequential([
        Embedding(input_dim=4, output_dim=4, input_length=input_length),
        LSTM(50, activation='tanh'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess data
filename = 'formal_language_dataset.csv'
samples, labels = load_dataset(filename)
X, y = preprocess_data(samples, labels, max_length=20)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train and evaluate the Vanilla RNN model
rnn_model = create_rnn_model(input_length=20)
rnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)

# Evaluate the RNN model
y_pred_rnn = (rnn_model.predict(X_test) > 0.5).astype("int32")
print("RNN Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rnn)}")
print(f"Precision: {precision_score(y_test, y_pred_rnn)}")
print(f"Recall: {recall_score(y_test, y_pred_rnn)}")
print(f"F1 Score: {f1_score(y_test, y_pred_rnn)}")

# Train and evaluate the LSTM model
lstm_model = create_lstm_model(input_length=20)
lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)

# Evaluate the LSTM model
y_pred_lstm = (lstm_model.predict(X_test) > 0.5).astype("int32")
print("LSTM Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lstm)}")
print(f"Precision: {precision_score(y_test, y_pred_lstm)}")
print(f"Recall: {recall_score(y_test, y_pred_lstm)}")
print(f"F1 Score: {f1_score(y_test, y_pred_lstm)}")
