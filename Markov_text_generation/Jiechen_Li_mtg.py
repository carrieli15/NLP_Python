import nltk
nltk.download('punkt')
nltk.download('gutenberg')
import numpy as np
from collections import Counter
def count_all_Word(input_corpus):
    word_counts = {}
    for word in input_corpus:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    sorted_word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))
    return sorted_word_counts    
# form corpus build a n_gram dictionary and return the dictionary
def build_ngram_model(corpus,n):
    ngram_model = {}
    ngram_list = [tuple(corpus[i:i + n]) for i in range(len(corpus) - n + 1)]
    for ngram in ngram_list:
        tokens = tuple(ngram[:-1])
        next_word = ngram[-1]   # last word of ngram 
        if tokens in ngram_model:  
            ngram_model[tokens].append(next_word)  
        else:
            ngram_model[tokens] = [next_word]  
    return ngram_model    
#  genarate the next word , we need to attention the unigram and multigram
def generate_nextWord(input_last_token, input_corpus,input_n, input_ramdomize):
    key = tuple(input_last_token)
    if input_n!=1:
        input_dics = build_ngram_model(input_corpus,input_n)
        if key in input_dics:
            posible_tokens = input_dics[key]
            most_count = Counter(posible_tokens).most_common() 
            if input_ramdomize:
                next_token =  np.random.choice(posible_tokens)
            else:
                next_token = most_count[0][0]     
            return next_token
        else:
            return None
    else:       #stupit backoff
        count_dict = count_all_Word(input_corpus)   #(input_corpus,input_n)

        if input_ramdomize:
            choices = []
            for i in count_dict.keys():
                for j in range(count_dict[i]):
                    choices.append(i)
            return np.random.choice(choices)
        else:
            return max(count_dict, key=count_dict.get)
# In this function I append the next word to sentence, until sentence length 
# plus and equal 10 or last of sentence word is punctuation.
def finish_sentence(sentence,n,corpus,randomize = False):
    # ngram_model_dict = build_ngram_model(corpus,n)
    
    while (len(sentence) < 10): # or (sentence[-1] not in ('.','?','!')):    
        last_tokens =  sentence[-(n-1):]
        continuation = n
        next_word = None
        while next_word is None:
            last_tokens =  sentence[-(continuation-1):]
            next_word = generate_nextWord(last_tokens,corpus,continuation,randomize)
            continuation -= 1
        sentence.append(next_word)
        if (sentence[-1] in ('.','?','!')):
            break
    return sentence
# -----------------------test the marcov----------------------------------
if __name__ == '__main__':
    # test_sentence = ["she","was","not"]
    test_sentence = ["robot"]
    n = 3
    corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw("austen-sense.txt").lower())
    randomize = False 
    result_deterministic = finish_sentence(test_sentence,n,corpus,randomize)
    print(result_deterministic)