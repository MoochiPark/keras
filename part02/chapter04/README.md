

# Part 02. 딥러닝 개념잡기



## Chapter 04. 학습 조기종료 시키기

앞서 과적합이라는 것과 이를 방지하기 위해 조기 종료하는 시점에 대해 간단히 알아 보았다. 
이 장에서는 어떻게 케라스가 제공하는 기능으로 학습 중에 조기 종료시킬 수 있는 지 알아보자.



### 조기 종료 시키기

`EarlyStopping()` 는 더 이상 개선의 여지가 없을 때 학습을 종료시키는 콜백함수다. 
콜백함수는 어떤 함수를 수행 시 그 함수에서 내가 지정한 함수를 호출하는 것을 말하며, 여기서는 `fit()`  함수에서
`EarlyStopping()`이라는 콜백함수가 학습 과정 중에 매번 호출된다. 

```python
early_stopping = EarlyStopping()
model.fit(X_train, Y_train, nb_epoch= 1000, callbacks=[early_stopping])
```

에포크를 1000으로 지정해도 학습과정에서 콜백함수를 호출하여 해당 조건이 되면 학습을 조기 종료 시킨다.

> EarlyStopping 콜백함수에서 설정할 수 있는 인자

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0,
                             mode='auto')
```

- **monitor** : 관찰하고자 하는 항목입니다. 'val_loss'나 'val_accuracy'가 주로 사용된다.
- **min_delta** : 개선되고 있다는 판단을 위한 최소 변화량을 나타낸다. 변화량이 min_delta보다 작다면 없다고 판단한다.
- **patience** : 개선이 없는 에포크를 얼마나 기다릴 것인지를 지정한다.
- **verbose** :  얼마나 자세하게 정보를 표시할 것인가를 지정한다. (0, 1, 2)
- **mode** : 관찰 항목에 대해 개선이 없다고 판단하기 위한 기준을 지정한다.
  - auto : 관찰하는 이름에 따라 자동으로 지정한다.
  - min : 관찰하고 있는 항목이 감소되는 것을 멈출 때 종료한다.
  - max : 관찰하고 있는 항목이 증가되는 것을 멈출 때 종료한다.



```python
# 모델 학습시키기
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping() # 조기종료 콜백함수 정의
hist = model.fit(X_train, Y_train, epochs=3000, batch_size=10, 
                 validation_data=(X_val, Y_val), callbacks=[early_stopping])
```

```python
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
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping() # 조기종료 콜백함수 정의
hist = model.fit(X_train, Y_train, epochs=3000, batch_size=10, validation_data=(X_val, Y_val), callbacks=[early_stopping])

# 5. 모델 학습 과정 표시하기
%matplotlib inline
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
```

> 실행 결과

```python
Epoch 38/3000
700/700 [==============================] - 0s 151us/step - loss: 1.3757 - accuracy: 0.4571 - val_loss: 1.4637 - val_accuracy: 0.3933
Epoch 39/3000
700/700 [==============================] - 0s 161us/step - loss: 1.3673 - accuracy: 0.4671 - val_loss: 1.4617 - val_accuracy: 0.4133
Epoch 40/3000
700/700 [==============================] - 0s 147us/step - loss: 1.3604 - accuracy: 0.4686 - val_loss: 1.4494 - val_accuracy: 0.4200
Epoch 41/3000
700/700 [==============================] - 0s 159us/step - loss: 1.3526 - accuracy: 0.4557 - val_loss: 1.4476 - val_accuracy: 0.4033
Epoch 42/3000
700/700 [==============================] - 0s 155us/step - loss: 1.3448 - accuracy: 0.4743 - val_loss: 1.4499 - val_accuracy: 0.4100
```



![image](https://user-images.githubusercontent.com/43429667/79750565-20d7fb00-834c-11ea-814c-4d128395c20f.png)

> 모델 사용하기

```python
   32/10000 [..............................] - ETA: 0s
loss : 1.439551894
accuray : 0.4443
```

val_loss 값이 감소되다가 증가하자마자 학습이 종료되었다. 하지만 val_loss의 특성상 증가/감소를 반복하므로 바로
종료하는 것이 아니라 지속적으로 증가되는 시점에 종료하도록 하자. 

```python
early_stopping = EarlyStopping(patience = 20)
```

![image](https://user-images.githubusercontent.com/43429667/79750870-a491e780-834c-11ea-81fa-e92e9e62cb92.png)

> 모델 사용하기

```python
   32/10000 [..............................] - ETA: 0s
loss : 1.34829078026
accuray : 0.5344
```

모델의 정확도도 향상됨을 확인할 수 있다.



### 요약

![image](https://user-images.githubusercontent.com/43429667/79751043-e327a200-834c-11ea-9c65-78eff9ccb88a.png)

