import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from common.util import clip_grads

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = list()
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # shuffle
            idx = np.random.permutation(np.arange(data_size))
            x, t = x[idx], t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # update weights
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads) # 공유된 가중치를 하나로 모음
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # evaluate
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print('Epoch: {:03} | Iteration: {} / {} | Time: {}s | Loss: {:0.2f}'.format(self.current_epoch+1, iters+1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('Iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('Loss')
        plt.show()

def remove_duplicate(params, grads):
    '''
    매개변수 배열 중 중복되는 가중치를 하나로 모아 그 가중치에 대응하는 기울기를 더한다
    '''
    params, grads = params[:], grads[:] # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L-1):
            for j in range(i+1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우 (weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                    np.transpose(params[i]).shape == params[j].shape and np.all(np.transpose(params[i]) == params[j]):
                    grads[i] += np.transpose(grads[j])
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break
        if not find_flg: break

    return params, grads