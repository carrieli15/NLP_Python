"""Compare token/document vectors for classification."""
import random
from typing import List, Mapping, Optional, Sequence

# import gensim
import nltk
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
import gensim.downloader as api


FloatArray = NDArray[np.float64]

# Un-comment this to fix the random seed
random.seed(31)

austen = nltk.corpus.gutenberg.sents("austen-sense.txt")
carroll = nltk.corpus.gutenberg.sents("carroll-alice.txt")
vocabulary = sorted(
    set(token for sentence in austen + carroll for token in sentence)
) + [None]
vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}


def onehot(
    vocabulary_map: Mapping[Optional[str], int], token: Optional[str]
) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary_map),))
    idx = vocabulary_map.get(token, len(vocabulary_map) - 1)
    embedding[idx] = 1
    return embedding


def sum_token_embeddings(
    token_embeddings: Sequence[FloatArray],
) -> FloatArray:
    """Sum the token embeddings."""
    total: FloatArray = np.array(token_embeddings).sum(axis=0)
    return total


def split_train_test(
    X: FloatArray, y: FloatArray, test_percent: float = 10
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Split data into training and testing sets."""
    N = len(y)
    data_idx = list(range(N))
    random.shuffle(data_idx)
    break_idx = round(test_percent / 100 * N)
    training_idx = data_idx[break_idx:]
    testing_idx = data_idx[:break_idx]
    X_train = X[training_idx, :]
    y_train = y[training_idx]
    X_test = X[testing_idx, :]
    y_test = y[testing_idx]
    return X_train, y_train, X_test, y_test


def generate_data_token_counts(
    h0_documents: list[list[str]], h1_documents: list[list[str]]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with raw token counts."""
    X: FloatArray = np.array(
        [
            sum_token_embeddings([onehot(vocabulary_map, token) for token in sentence])
            for sentence in h0_documents
        ]
        + [
            sum_token_embeddings([onehot(vocabulary_map, token) for token in sentence])
            for sentence in h1_documents
        ]
    )
    y: FloatArray = np.array(
        [0 for sentence in h0_documents] + [1 for sentence in h1_documents]
    )
    return split_train_test(X, y)


def generate_data_tfidf(
    h0_documents: list[list[str]], h1_documents: list[list[str]]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with TF-IDF scaling."""
    X_train, y_train, X_test, y_test = generate_data_token_counts(
        h0_documents, h1_documents
    )
    tfidf = TfidfTransformer(norm=None).fit(X_train)
    X_train = tfidf.transform(X_train)
    X_test = tfidf.transform(X_test)
    return X_train, y_train, X_test, y_test


def generate_data_lsa(
    h0_documents: list[list[str]], h1_documents: list[list[str]]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with LSA."""
    X_train, y_train, X_test, y_test = generate_data_token_counts(
        h0_documents, h1_documents
    )
    #########################################################################
    # DO SOMETHING HERE.                                                    #
    #########################################################################
    # Step1: Generate data using token counts
    X_train, y_train, X_test, y_test = generate_data_token_counts(
        h0_documents, h1_documents
    )
    # Step2: Apply LSA transformation
    num_components = 300  # define the number of components for LSA
    lsa = TruncatedSVD(n_components=num_components, n_iter=7, random_state=42)
    # Fit and transform LSA on training data
    X_train_lsa = lsa.fit_transform(X_train)
    # Transform LSA on test data
    X_test_lsa = lsa.transform(X_test)

    return X_train_lsa, y_train, X_test_lsa, y_test


def generate_data_word2vec(
    h0_documents: list[list[str]], h1_documents: list[list[str]]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with word2vec."""
    #########################################################################
    # DO SOMETHING HERE.                                                    #
    #########################################################################
    # Step 1: Train Word2Vec model
    # word2vector_model = train_word2vec_model(h0_documents + h1_documents)
    pretrained_word2vec_model = api.load("glove-wiki-gigaword-100")
    # Step 2: Calculate document vectors
    h0_vectors = calculate_document_vectors(h0_documents, pretrained_word2vec_model)
    h1_vectors = calculate_document_vectors(h1_documents, pretrained_word2vec_model)
    # Step 3: Assign labels
    h0_labels = np.zeros(len(h0_vectors))
    h1_labels = np.ones(len(h1_vectors))
    # Concatenate vectors and labels
    X = np.concatenate([h0_vectors, h1_vectors], axis=0)
    y = np.concatenate([h0_labels, h1_labels], axis=0)
    # Return the split_train_test function
    return split_train_test(X, y, 10)


def run_experiment() -> None:
    """Compare performance with different embeddiings."""
    X_train, y_train, X_test, y_test = generate_data_token_counts(austen, carroll)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    print("raw counts (train):", clf.score(X_train, y_train))
    print("raw_counts (test):", clf.score(X_test, y_test))

    X_train, y_train, X_test, y_test = generate_data_tfidf(austen, carroll)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    print("tfidf (train):", clf.score(X_train, y_train))
    print("tfidf (test):", clf.score(X_test, y_test))

    X_train, y_train, X_test, y_test = generate_data_lsa(austen, carroll)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    print("lsa (train):", clf.score(X_train, y_train))
    print("lsa (test):", clf.score(X_test, y_test))

    X_train, y_train, X_test, y_test = generate_data_word2vec(austen, carroll)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    print("word2vec (train):", clf.score(X_train, y_train))
    print("word2vec (test):", clf.score(X_test, y_test))


# def train_word2vec_model(documents):
#     model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=1, workers=4)
#     return model


def calculate_document_vectors(documents, word2vec_model):
    document_vectors = []  # List to store document vectors

    for doc in documents:  # For each document in the input documents
        word_vectors = []  # List to store word vectors

        for word in doc:  # For each word in the document
            if (
                word in word2vec_model
            ):  # If the word is in the Word2Vec model's vocabulary
                word_vector = word2vec_model[word]  # Get the word vector for the word
                word_vectors.append(word_vector)  # Add the word vector to the list

        if word_vectors:  # If there are word vectors in the document
            doc_vector = np.mean(
                word_vectors, axis=0
            )  # Calculate the document vector by averaging word vectors
            document_vectors.append(doc_vector)  # Add the document vector to the list

    return np.array(document_vectors)  # Return document vectors as a NumPy array


if __name__ == "__main__":
    run_experiment()
