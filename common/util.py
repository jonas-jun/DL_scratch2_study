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

def create_co_matrix(corpus, vocab_size, window_size=1):
    '''
    create matrix of words representation which contains near words (동시발생행렬)
    param corpus: 말뭉치 (단어 ID 목록)
    param vocab_size: 어휘 수
    param window_size: 윈도 크기 (1일 때 좌우 한 단어씩 맥락에 포함)
    return: 동시발생 행렬
    '''
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''
    params
    - query: text
    - word_to_id, id_to_word: dictionary
    - word_matrix: 단어 벡터를 정리한 행렬, 각 행에 각 단어벡터가 저장되어 있음
    - top: the number of return words
    output
    - 유사단어(들) 출력 (NOT return BUT print)
    '''

    if query not in word_to_id:
        raise ValueError ('cannot find {} in input words'.format(query))

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    
    # 코사인 유사도 기준으로 내림차순 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' {}: {}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

def ppmi(C, verbose=False, eps=1e-8):
    '''
    PPMI(점별 상호정보량) 생성
    params
    - C: 동시발생 행렬
    - verbose: 진행상황 출력여부
    '''

    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print('{:0.1f}% 완료'.format(100*cnt/total))
    
    return M