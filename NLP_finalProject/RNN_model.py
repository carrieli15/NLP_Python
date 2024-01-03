import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# from lib.sythetic_data import *
from lib.UnigramModel import UnigramModel

from lib.ProcessEmail import process_email_body_simple
from lib.NaiveBayesEmailClassifier import NaiveBayesEmailClassifier
from get_info import *


def RNN_main(data):
    df = data
    # Convert to string and handle NaNs
    df["Body"] = df["Body"].fillna("").astype(str)
    # Tokenization
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["Body"])

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df["Body"], df["Label"], test_size=0.2, random_state=42
    )
    train_sequences = tokenizer.texts_to_sequences(X_train)
    # train Padding
    train_padded_sequences = pad_sequences(train_sequences, maxlen=200)

    # # Labels
    train_labels = df["Label"].values
    model = Sequential()
    model.add(Embedding(5000, 64, input_length=200))
    model.add(SimpleRNN(64))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    # train the model
    model.fit(train_padded_sequences, train_labels, epochs=10, validation_split=0.2)
    # test the model
    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_padded_sequences = pad_sequences(test_sequences, maxlen=200)
    test_labels = y_test.values
    loss, accuracy = model.evaluate(test_padded_sequences, test_labels)
    print(f"Test Accuracy: {accuracy}")


if __name__ == "__main__":
    # Load dataset
    data_file = "/workspaces/NLP_finalProject/data/completeSpamAssassin.csv"
    df_original = pd.read_csv(data_file)
    RNN_main(df_original)

    # for sythetic dataset
    sythetic_data_file = "/workspaces/NLP_finalProject/data/sythetic_dataset.csv"
    df_sythetic = pd.read_csv(sythetic_data_file)
    RNN_main(df_sythetic)
