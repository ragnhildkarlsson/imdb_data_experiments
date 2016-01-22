import index

def print_stop_word_list_to_file(index, stop_word_file_path):
    n_words_in_corpus = sum([sum([post[1] for post in index[word]]) for word in index])
    print("n_words_in_corpus calcullated")
    stop_words = []
    for word in index:
        word_freq = sum([post[1] for post in index[word]]) 
        if (word_freq/n_words_in_corpus) > 0.0011:
            stop_words.append(word)
    print('word_freq_calculated')
    with open(stop_word_file_path, 'w') as f:
        for stop_word in stop_words:
                f.write(stop_word+'\n')

def print_words_high_doc_frequency(index, doc_frequency_limit ,docs_in_corpus):
    frequent_words =set()
    for word in index:
        n_docs = len(index_word)
        if n_docs/docs_in_corpus > doc_frequency_limit:
            print word
            frequent_words.add(word)
