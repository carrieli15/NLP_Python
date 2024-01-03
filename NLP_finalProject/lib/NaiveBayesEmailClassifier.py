from typing import List, Tuple, Optional, Sequence
import numpy as np
import random
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class NaiveBayesEmailClassifier:
    def __init__(self, vocabulary_map):
        self.vocabulary_map = vocabulary_map
        self.h0_language_model = None
        self.h1_language_model = None
        self.ph0 = None
        self.ph1 = None

    def onehot(self, token: Optional[str]) -> np.ndarray:
        """Generate the one-hot encoding for the provided token."""
        embedding = np.zeros((len(self.vocabulary_map), 1))
        idx = self.vocabulary_map.get(token, len(self.vocabulary_map) - 1)
        embedding[idx, 0] = 1
        return embedding

    def encode_document(self, tokens: Sequence[Optional[str]]) -> List[np.ndarray]:
        """Apply one-hot encoding to each document."""
        return [self.onehot(token) for token in tokens]

    def split_data(
        self, sample_spam: List[str], sample_nonspam: List[str], test_percent: int = 10
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Assemble the training and testing data from spam and non-spam samples."""
        h0_observations = [
            (self.encode_document(sentence), 0) for sentence in sample_spam
        ]
        h1_observations = [
            (self.encode_document(sentence), 1) for sentence in sample_nonspam
        ]

        all_data = h0_observations + h1_observations
        random.shuffle(all_data)

        break_idx = round(len(all_data) * test_percent / 100)
        return all_data[break_idx:], all_data[:break_idx]

    def train_naive_bayes(self, training_data, UnigramModel):
        """Train the Naive Bayes model using the training data."""
        h0_documents = [
            observation[0] for observation in training_data if observation[1] == 0
        ]
        h1_documents = [
            observation[0] for observation in training_data if observation[1] == 1
        ]

        self.h0_language_model = UnigramModel(len(self.vocabulary_map))
        self.h0_language_model.train(
            [token for document in h0_documents for token in document]
        )

        self.h1_language_model = UnigramModel(len(self.vocabulary_map))
        self.h1_language_model.train(
            [token for document in h1_documents for token in document]
        )

        self.ph0 = len(h0_documents) / len(training_data)
        self.ph1 = len(h1_documents) / len(training_data)

    def predict(self, document):
        """Predict the label of a given document."""
        h0_logp_unnormalized = self.h0_language_model.apply(document) + np.log(self.ph0)
        h1_logp_unnormalized = self.h1_language_model.apply(document) + np.log(self.ph1)

        logp_data = np.logaddexp(h0_logp_unnormalized, h1_logp_unnormalized)
        h0_logp = h0_logp_unnormalized - logp_data
        h1_logp = h1_logp_unnormalized - logp_data

        pc0 = np.exp(h0_logp)
        pc1 = np.exp(h1_logp)
        return 1 if pc1 > pc0 else 0
