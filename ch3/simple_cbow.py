import sys
sys.path.append('/Users/jonas/dl_scratch2')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # instantiate weights
        w_in = 0.01 * np.random.randn(V, H).astype('f')
        w_out = 0.01 * np.random.randn(H, V).astype('f')

        # build layers
        self.in_layer0 = MatMul(w_in)
        self.in_layer1 = MatMul(w_in)
        self.out_layer = MatMul(w_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 분산 표현 저장
        self.word_vecs = w_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0]) # 앞 인접 단어, mini_batch * hidden_size
        h1 = self.in_layer1.forward(contexts[:, 1]) # 뒤 인접 단어, mini_batch * hidden_size
        # print(h0.shape, h1.shape) -> matmul 에서 W, 대신 W를 넣었을 시 (3,5)가 아닌 (3,1,5)가 출력됨.
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h) # mini_batch * vocab_size
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
