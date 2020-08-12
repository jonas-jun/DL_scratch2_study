# DL_scratch2_study
practice manual 'deep learning from scratch2', '밑바닥부터 시작하는 딥러닝 2'

## Deep Learning from Scratch2 (밑바닥부터 시작하는 딥러닝2)

ch1. 신경망 기초
  - forward, backward 포함한 class 구현
  - loss와 weights update

ch2. 통계적 자연어 처리 기법
  - C(동시발생행렬): 함께 나오는 단어들을 window size 만큼 살펴보고 벡터로 표현
  - W(상호정보량): 'the'가 'car'하고만 많이 나오나? 아니다. 상대적으로 함께 자주 나오는 단어들을 표기하는 기법
  - SVD(차원축소): sparse vector가 아닌 dense vector로 표기
  
ch3. Word2Vec 1 (추론적 처리 기법)
  - CBOW: 주변 단어들을 사용하여 target word를 추정할 수 있도록 represent
      input으로 주변 여러 단어들을 받고 각각 동일한 weight(in)을 곱해준 다음, 하나의 단어 벡터를 뽑아서 target과 비교
  - skip-gram: 하나의 단어를 받아서 주변 단어들을 추정하는 방식
      input을 하나의 단어 벡터로 받고, hidden_layer를 지나 여러 주변 단어 벡터들로 출력
      이때 loss는 각 모든 단어들의 loss들의 총합 (p.144)
  - 학습 속도는 CBOW가 빠르지만 성능은 skip-gram이 좋은 것으로 알려져 있다. 손실을 구해야할 벡터들이 많기 때문에 CBOW의 속도가 느리다.

ch4. Word2Vec 2 (embedding & negative sampling)
  - 단어 수가 급수적으로 늘어날 때 2곳에서 병목이 발생할 수 있다. 1)input * W = hidden, 2)softmax(score)
  - embedding: 첫 번째 bottleneck을 해결해준다. W층에서 필요한 index의 rows만 가져온다. (구현은 layers에)
  