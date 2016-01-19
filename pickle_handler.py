import pickle

def load_pickle(file_path):
    resourse = pickle.load( open(file_path, "rb" ) )
    return resourse

def print_pickle(resourse, file_path):
    pickle.dump(resourse, open(file_path,'wb'))
