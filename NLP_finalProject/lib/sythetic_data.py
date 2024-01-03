# this code is to generate sythetic email data
import pandas as pd
import random
from ProcessEmail import process_email_body_simple

# from sythetic_data import *

# import packages from Generative_Probabilistic_Model
# sys.path.append(os.path.abspath(os.path.join("..")))
# from Generative_Probabilistic_Model.ProcessEmail import process_email_body_simple


# define a function that create dictionaries for spam and non-spam that records the frequency of each word
def create_dict(vocabulary, emails):
    dict = {}
    for word in vocabulary:
        dict[word] = 1
    for lines in emails:
        for word in lines:
            dict[word] += 1
    return dict


def select_next(vocab_freq):
    # print("stochastic running")
    weight = [val for val in vocab_freq.values()]
    ls = [e for e in vocab_freq.keys()]
    return random.choices(ls, weight, k=1).pop()


def generate_sythetic(original_data, vocabulary):
    nlines = len(original_data)
    vocab_freq = create_dict(vocabulary, original_data)
    sythetic_data = []
    for _ in range(nlines):
        line = []
        while len(line) < 15:
            next_word = select_next(vocab_freq)
            line.append(next_word)
        sythetic_data.append(line)
    return sythetic_data


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv("/workspaces/NLP_finalProject/data/completeSpamAssassin.csv")

    # Splitting the dataframe into two based on the label
    spam_df = df[df["Label"] == 1]
    non_spam_df = df[df["Label"] == 0]

    # Applying the simplified processing function to each group
    spam_emails = spam_df["Body"].apply(process_email_body_simple)
    non_spam_emails = non_spam_df["Body"].apply(process_email_body_simple)

    # Compile the processed emails into separate lists for spam and non-spam
    spam_list = [line for email in spam_emails for line in email]
    non_spam_list = [line for email in non_spam_emails for line in email]

    # Display the first few elements of the compiled simple list to verify the structure
    sample_spam = spam_list[:800]  # 300
    sample_nonspam = non_spam_list[:1000]  # 500
    # print(sample_spam[0])
    # print(sample_spam)
    vocabulary = sorted(
        set(token for sentence in sample_spam + sample_nonspam for token in sentence)
    )  # + [None]
    spam_sythetic = generate_sythetic(sample_spam, vocabulary)
    nonspam_sythetic = generate_sythetic(sample_nonspam, vocabulary)

    # for each row in spam_sythetic, join the words in the row with space
    spam_sythetic_join = [" ".join(row) for row in spam_sythetic]
    # store the list in a dataframe in a column named "Body", and add a column named "Label" with value 1
    spam_sythetic_df = pd.DataFrame(spam_sythetic_join, columns=["Body"])
    spam_sythetic_df["Label"] = 1

    # perform the same operation for nonspam_sythetic
    nonspam_sythetic_join = [" ".join(row) for row in nonspam_sythetic]
    nonspam_sythetic_df = pd.DataFrame(nonspam_sythetic_join, columns=["Body"])
    nonspam_sythetic_df["Label"] = 0

    # concatenate spam_sythetic_df and nonspam_sythetic_df
    sythetic_df = pd.concat([spam_sythetic_df, nonspam_sythetic_df])
    # save the dataframe as a csv file
    sythetic_df.to_csv(
        "/workspaces/NLP_finalProject/data/sythetic_dataset.csv", index=False
    )
