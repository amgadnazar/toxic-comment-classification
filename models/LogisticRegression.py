"""
Logistic Regression baseline for toxic comment classification
Using TF-IDF features.

Dataset: Jigsaw Toxic Comment Classification
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# =========================
# Data Loading
# =========================

TRAIN_PATH = r"YOUR_PATH\jigsaw-toxic-comment-classification-challenge\train.csv"
TEST_PATH = r"YOUR_PATH\jigsaw-toxic-comment-classification-challenge\test.csv"
TEST_LABELS_PATH = r"YOUR_PATH\jigsaw-toxic-comment-classification-challenge\test_labels.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
test_labels = pd.read_csv(TEST_LABELS_PATH)

# Merge test data with labels
test_df = test_df.merge(test_labels, on="id")

# Remove samples with no labels (-1)
test_df = test_df[test_df['toxic'] != -1]


# =========================
# Label Creation
# =========================

def create_label(df):
    """
    Convert multi-label toxicity columns into a single binary label.
    1 -> toxic (if any category is present)
    0 -> non-toxic
    """
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
# Feature Extraction (TF-IDF)
# =========================

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words='english'
)

X_train = vectorizer.fit_transform(train_df['comment_text'])
X_test = vectorizer.transform(test_df['comment_text'])

y_train = train_df['label']
y_test = test_df['label']


# =========================
# Model Training
# =========================

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'  # Handle class imbalance
)

model.fit(X_train, y_train)


# =========================
# Evaluation Function
# =========================

def evaluate_threshold(y_true, y_probs, threshold):
    """
    Convert probabilities to predictions using a given threshold
    and print evaluation metrics.
    """
    y_pred = (y_probs > threshold).astype(int)

    print(f"\nThreshold = {threshold}")
    print(classification_report(y_true, y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


# =========================
# Prediction + Evaluation
# =========================

y_probs = model.predict_proba(X_test)[:, 1]

# Evaluate different thresholds
for threshold in [0.50, 0.70, 0.75]:
    evaluate_threshold(y_test, y_probs, threshold)