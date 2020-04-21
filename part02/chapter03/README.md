# Part 02. 딥러닝 개념잡기



## Chapter 03. 학습과정 표시하기<sup>텐서보드 포함</sup>

케라스로 딥러닝 모델을 개발할 때, 가장 많이 보게 되는 것이 `fit()` 함수로 화면에 찍는 로그이다. 
이 로그에 포함된 수치가 학습이 제대로 되고 있는지, 학습을 그만할 지 등을 판단하는 중요한 척도가 된다.

수치들이 에포크마다 바뀌는 변화 추이를 보는 것이 중요하므로 그래프로 표시하여 보는 것이 더 직관적인데,
다음과 같은 방법으로 알아볼 것이다. 

- 케라스에서 제공하는 기능 - 히스토리 기능 사용하기
- 직접 콜백함수를 만들어 보는 방법



### 히스토리 기능 사용하기

케라스의 `fit()` 함수의 반환 값으로 히스토리 객체를 얻을 수 있는데, 이 객체는 다음의 정보를 담고 있다.

- 매 에포크 마다의 훈련 손실값 (loss)
- 매 에포크 마다의 훈련 정확도 (accuracy)
- 매 에포크 마다의 검증 손실값 (val_loss)
- 매 에포크 마다의 검증 정확도 (val_acc)



히스토리 기능은 케라스의 모든 모델에 탑재되어 있으므로 별도의 설정 없이 다음처럼 사용할 수 있다.

```python
hist = model.fit(X_train, Y_train, epochs=1000, batch_size=10, 
                 validation_data=(X_val, Y_val))

print(hist.history['loss'])
print(hist.history['acc'])
print(hist.history['val_loss'])
print(hist.history['val_acc'])
```

수치들은 각 에포크마다 해당 값이 추가되므로 리스트 형태로 되어있다. 

아래와 같이 matplotlib 패키지를 이용하면 하나의 그래프로 표시할 수 있다.

<script src="https://gist.github.com/MoochiPark/0fc15eb765a6b0d4b20a3b43047c7c4a.js"></script>

각 에포크에 대한 손실값, 정확도 추이를 볼 수 있다. 검증셋의 손실값이 100번째 에포크에서 다시 증가되는 것을 볼 수 있는데,
과적합<sup>overfitting</sup>이 발생했다고 볼 수 있다. 이 경우 100번째 에포크까지 학습시킨 모델이 1000번 학습시킨 모델보다 좋은 결과가 나올 수 있다.

----

### 직접 콜백함수 만들어보기

기본적인 모델의 학습 상태 모니터링은 히스토리 콜백함수 또는 텐서보드를 사용하면 되지만,
순환신경망<sup>RNN</sup> 모델인 경우에는 `fit()` 함수를 여러번 호출되기 때문에 제대로 학습상태를 볼 수 없다.

> *순환 신경망 코드*

```python
for epoch_idx in range(1000):
    print ('epochs : ' + str(epoch_idx) )
    hist = model.fit(train_X, train_Y, epochs=1, batch_size=1, 
                     verbose=2, shuffle=False) # 50 is X.shape[0]
    model.reset_states()
```

매 에포크마다 히스토리 객체가 생성되어 매번 초기화 되기 때문에 에포크별로 추이를 볼 수 없다. 이 문제를 해결하기 위해
`fit()` 함수를 여러 번 호출되더라도 학습 상태가 유지될 수 있도록 콜백함수를 정의해보자.

```python
class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []        
        
    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
```

```python
import keras

# 사용자 정의 히스토리 클래스 정의
class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []        
        
    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))
    
# 모델 학습시키기

from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

np.random.seed(3)

# 1. 데이터셋 준비하기

# 훈련셋과 시험셋 로딩
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 훈련셋과 검증셋 분리
X_val = X_train[50000:]
Y_val = Y_train[50000:]
X_train = X_train[:50000]
Y_train = Y_train[:50000]

X_train = X_train.reshape(50000, 784).astype('float32') / 255.0
X_val = X_val.reshape(10000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0

# 훈련셋, 검증셋 고르기
train_rand_idxs = np.random.choice(50000, 700)
val_rand_idxs = np.random.choice(10000, 300)

X_train = X_train[train_rand_idxs]
Y_train = Y_train[train_rand_idxs]
X_val = X_val[val_rand_idxs]
Y_val = Y_val[val_rand_idxs]

# 라벨링 전환
Y_train = np_utils.to_categorical(Y_train)
Y_val = np_utils.to_categorical(Y_val)
Y_test = np_utils.to_categorical(Y_test)

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=2, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 엮기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기

custom_hist = CustomHistory()
custom_hist.init()

for epoch_idx in range(1000):
    print ('epochs : ' + str(epoch_idx) )
    model.fit(X_train, Y_train, epochs=1, batch_size=10, validation_data=(X_val, Y_val), callbacks=[custom_hist])

# 5. 모델 학습 과정 표시하기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(custom_hist.train_loss, 'y', label='train loss')
loss_ax.plot(custom_hist.val_loss, 'r', label='val loss')

acc_ax.plot(custom_hist.train_acc, 'b', label='train acc')
acc_ax.plot(custom_hist.val_acc, 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
```

