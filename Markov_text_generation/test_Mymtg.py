from mtg import finish_sentence
import nltk
# -----------------------test the marcov----------------------------------
if __name__ == '__main__':
    # test_centence = ["she","was","not"]
    test_centence = ["robot"]
    n = 3
    corpus = nltk.word_tokenize(
    nltk.corpus.gutenberg.raw("austen-sense.txt").lower())
    randomize = False 
    result_deterministic = finish_sentence(test_centence,n,corpus,randomize)
    print(result_deterministic)

