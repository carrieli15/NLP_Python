"""Demonstrate Naive Bayes classification."""
from lib.UnigramModel import UnigramModel

from lib.ProcessEmail import process_email_body_simple
from lib.NaiveBayesEmailClassifier import NaiveBayesEmailClassifier
from get_info import *
from lib.sythetic_data import *


def main(sample_spam, sample_nonspam, vocabulary_map):
    # Initialize the classifier with the vocabulary map
    classifier = NaiveBayesEmailClassifier(vocabulary_map)

    # Assume UnigramModel is defined elsewhere
    # Train the classifier
    training_data, testing_data = classifier.split_data(sample_spam, sample_nonspam)
    classifier.train_naive_bayes(training_data, UnigramModel)

    # Test the classifier
    num_correct = 0
    for document, label in testing_data:
        guess = classifier.predict(document)
        # print(guess, label)  # Optional: Print prediction and actual label

        if guess == label:
            num_correct += 1

    # Print the accuracy
    print(f"Accuracy: {num_correct / len(testing_data)}")


if __name__ == "__main__":
    file_path = "/workspaces/NLP_finalProject/data/completeSpamAssassin.csv"
    spam_size, nonspam_size = (800, 1000)
    sample_spam, sample_nonspam, vocabulary_map = get_info(
        file_path, spam_size, nonspam_size
    )
    print("model accuracy on original dataset:")
    main(sample_spam, sample_nonspam, vocabulary_map)

    # =============================================================
    # generate sythetic data
    # =============================================================
    # sys.path.append("/workspaces/NLP_finalProject/Generative_Probabilistic_Model")

    spam_sythetic = generate_sythetic(sample_spam, vocabulary_map.keys())
    nonspam_sythetic = generate_sythetic(sample_nonspam, vocabulary_map.keys())
    print("model accuracy on sythetic dataset:")
    main(spam_sythetic, nonspam_sythetic, vocabulary_map)
