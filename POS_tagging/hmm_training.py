import numpy as np
import nltk
from viterbi import *


# nltk.download("brown")
# nltk.download("universal_tagset")

SMOOTHING = 0.7


def initiate_state(tagged_sents):
    tag_frequencies = {}
    for sent in tagged_sents:
        tag = sent[0][1]
        tag_frequencies[tag] = tag_frequencies.get(tag, 0) + 1

    total_sents = len(tagged_sents)
    return {tag: freq / total_sents for tag, freq in tag_frequencies.items()}


def calculate_frenq(tagged_sents):
    tag_pairs = {}
    for sent in tagged_sents:
        for i in range(len(sent) - 1):
            pair = (sent[i][1], sent[i + 1][1])
            tag_pairs[pair] = tag_pairs.get(pair, 0) + 1
    return tag_pairs


def transition(initial_state, tag_pair_freq):
    tags = list(initial_state.keys())
    matrix_size = len(tags)
    matrix = np.full((matrix_size, matrix_size), SMOOTHING)

    for i, tag1 in enumerate(tags):
        for j, tag2 in enumerate(tags):
            matrix[i, j] = tag_pair_freq.get((tag1, tag2), 0)

    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def emission(initial_state, tagged_sents):
    unique_words = list(set(word for sent in tagged_sents for word, _ in sent))
    unique_words.append("OOV")

    tags = list(initial_state.keys())
    emission_matrix = np.full((len(tags), len(unique_words)), SMOOTHING)
    word_indices = {word: idx for idx, word in enumerate(unique_words)}
    tag_indices = {tag: idx for idx, tag in enumerate(tags)}

    for sent in tagged_sents:
        for word, tag in sent:
            word_idx = word_indices.get(word, word_indices["OOV"])
            tag_idx = tag_indices[tag]
            emission_matrix[tag_idx, word_idx] += 1

    emission_matrix /= emission_matrix.sum(axis=1, keepdims=True)
    return emission_matrix


def observation_seq(test_sents, tagged_sents):
    unique_words = list(set(word for sent in tagged_sents for word, _ in sent))
    unique_words.append("OOV")
    word_indices = {word: idx for idx, word in enumerate(unique_words)}

    return [
        word_indices.get(word[0], word_indices["OOV"])
        for sent in test_sents
        for word in sent
    ]


def tag(test_sents, initial_state):
    tags = list(initial_state.keys())
    return [tags.index(word[1]) for sent in test_sents for word in sent]


if __name__ == "__main__":
    tagged_sents = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
    test_sents = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]

    tag_initial_state = initiate_state(tagged_sents)
    pi = list(tag_initial_state.values())
    tag_pair_freq = calculate_frenq(tagged_sents)
    transmaxtrix = transition(tag_initial_state, tag_pair_freq)
    emissmatrix = emission(tag_initial_state, tagged_sents)
    obs = observation_seq(test_sents, tagged_sents)
    model_tag_seq = viterbi(obs, pi, transmaxtrix, emissmatrix)
    true_tag_seq = tag(test_sents, tag_initial_state)

    print("model prediction:")
    print(model_tag_seq[0])
    print("true results")
    print(true_tag_seq)
    res = [a == b for a, b in zip(model_tag_seq[0], true_tag_seq)]
    print("accuracy of the model is:")
    print(sum(res) / len(res))
