import numpy as np

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad += rate

def preprocess(text):
    '''
    make a corpus dictionary, and integer vectors
    '''
    text = text.lower()
    text = text.replace('.', ' .') # 마침표를 띄워주기 위함
    words = text.split(' ') # 단어 단위로 쪼갠 list

    word_to_id = dict()
    id_to_word = dict()
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word
