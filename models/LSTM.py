"""
Bidirectional LSTM model for toxic comment classification.

Pipeline:
Text -> Tokenizer -> Padding -> Embedding -> BiLSTM -> Classification
"""

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================
# Data Loading
# =========================

TRAIN_PATH = r"YOUR_PATH\jigsaw-toxic-comment-classification-challenge\train.csv"
TEST_PATH = r"YOUR_PATH\jigsaw-toxic-comment-classification-challenge\test.csv"
TEST_LABELS_PATH = r"YOUR_PATH\jigsaw-toxic-comment-classification-challenge\test_labels.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
test_labels = pd.read_csv(TEST_LABELS_PATH)

test_df = test_df.merge(test_labels, on="id")
test_df = test_df[test_df['toxic'] != -1]


# =========================
# Label Creation
# =========================

def create_label(df):
    df['label'] = (
        df['toxic'] +
        df['severe_toxic'] +
        df['obscene'] +
        df['threat'] +
        df['insult'] +
        df['identity_hate']
    )
    df['label'] = df['label'].apply(lambda x: 1 if x > 0 else 0)
    return df[['comment_text', 'label']]


train_df = create_label(train_df)
test_df = create_label(test_df)


# =========================
# Tokenization
# =========================

max_words = 20000
max_len = 150

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['comment_text'])

X_train = tokenizer.texts_to_sequences(train_df['comment_text'])
X_test = tokenizer.texts_to_sequences(test_df['comment_text'])

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

y_train = train_df['label']
y_test = test_df['label']


# =========================
# Class Weights (Handle Imbalance)
# =========================

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {0: weights[0], 1: weights[1]}


# =========================
# Model Definition
# =========================

model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


# =========================
# Training
# =========================

history = model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=32,
    class_weight=class_weights,
    validation_split=0.1
)


# =========================
# Evaluation Function
# =========================

def evaluate_threshold(y_true, y_probs, threshold):
    y_pred = (y_probs > threshold).astype(int)

    print(f"\nThreshold = {threshold}")
    print(classification_report(y_true, y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


# =========================
# Prediction + Evaluation
# =========================

y_probs = model.predict(X_test).flatten()

for threshold in [0.75, 0.70, 0.50]:
    evaluate_threshold(y_test, y_probs, threshold)