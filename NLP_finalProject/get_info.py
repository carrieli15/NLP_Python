# this code is to return email spam and non spam samples as well as the vocabulary
# from lib.ProcessEmail import process_email_body_simple
from lib.ProcessEmail import process_email_body_simple
import pandas as pd


# define a function that returns vocabulary_map
def get_vocabulary_map(vocabulary):
    vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}

    return vocabulary_map


def get_info(file_path, spam_size, nonspam_size):
    # Load the data
    df = pd.read_csv(file_path)

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
    sample_spam = spam_list[:spam_size]  # 300
    sample_nonspam = non_spam_list[:nonspam_size]  # 500

    vocabulary = sorted(
        set(token for sentence in sample_spam + sample_nonspam for token in sentence)
    ) + [None]

    # define vocabulary_map
    vocabulary_map = get_vocabulary_map(vocabulary)

    return sample_spam, sample_nonspam, vocabulary_map


if __name__ == "__main__":
    file_path = "/workspaces/NLP_finalProject/data/completeSpamAssassin.csv"
    spam_size, nonspam_size = (800, 1000)
    sample_spam, sample_nonspam, vocabulary_map = get_info(
        file_path, spam_size, nonspam_size
    )
