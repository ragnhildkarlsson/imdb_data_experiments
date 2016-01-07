import document_processer 
import index

WORD_INDEX_PICKLE_FILE = "data/index_word_pickle_file"
BIGRAM_INDEX_PICKLE_FILE = "data/index_bigram_pickle_file"
NUMBER_OF_DOCUMENTS = 220000


def print_index_statistics():
    word_index = index.get_index(WORD_INDEX_PICKLE_FILE)
    print('Number of term in corpus:')
    print(str(len(word_index)))
    max_len = 0
    for(word in word_index):
        if(len(word_index[word])>max_len):
            max_len = len(word_index[word])
    print('Most frequent term: '+ str(max_len))

    bigram_index = index.get_index(BIGRAM_INDEX_PICKLE_FILE)
    print('Number of bigram in corpus:')
    print(str(len(word_index)))
    

