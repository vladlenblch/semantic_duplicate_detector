import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from scipy.sparse import hstack


TARGET_COLUMN = "is_duplicate"
LOGREG_MODE = "logreg"
COSINE_MODE = "cosine"
VALID_MODES = {LOGREG_MODE, COSINE_MODE}


class TFIDFBaseline:
    def __init__(self, threshold=0.5):
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100_000,
            min_df=2,
            stop_words="english",
            norm="l2"
        )

        self.classifier = LogisticRegression(
            max_iter=10_000,
            class_weight="balanced"
        )

        self.mode = None
        self.threshold = threshold

    def fit(self, df, mode=LOGREG_MODE):
        self.mode = mode

        all_questions = pd.concat([
            df["question1"],
            df["question2"]
        ])

        self.tfidf_vectorizer.fit(all_questions)

        if self.mode == LOGREG_MODE:
            X = self._make_pair_features(df)
            y = df[TARGET_COLUMN]

            self.classifier.fit(X, y)

        return self

    def predict(self, df):
        if self.mode == COSINE_MODE:
            similarities = self.predict_proba(df)

            return (similarities >= self.threshold).astype(int)

        if self.mode == LOGREG_MODE:
            X = self._make_pair_features(df)

            return self.classifier.predict(X)
    
    def predict_proba(self, df):
        if self.mode == COSINE_MODE:
            return self._calculate_cosine_similarity(df)

        if self.mode == LOGREG_MODE:
            X = self._make_pair_features(df)

            return self.classifier.predict_proba(X)[:, 1]

    def _make_pair_features(self, df):
        q1_vectors, q2_vectors = self._make_question_vectors(df)

        diff_vectors = q1_vectors - q2_vectors
        diff_vectors.data = np.abs(diff_vectors.data)

        multiply_vectors = q1_vectors.multiply(q2_vectors)

        pair_features = hstack([
            diff_vectors,
            multiply_vectors,
        ])

        return pair_features

    def _calculate_cosine_similarity(self, df):
        q1_vectors, q2_vectors = self._make_question_vectors(df)

        similarities = q1_vectors.multiply(q2_vectors).sum(axis=1)
        similarities = np.asarray(similarities).ravel()

        return similarities

    def _make_question_vectors(self, df):
        q1_vectors = self.tfidf_vectorizer.transform(df["question1"])
        q2_vectors = self.tfidf_vectorizer.transform(df["question2"])

        return q1_vectors, q2_vectors
