# Part 02. 딥러닝 개념잡기



## Chapter 06. 학습 모델 보기/저장/불러오기

아직도 딥러닝 모델을 사용하려면 '매번 몇시간 학습시켜야 하는 것인가?' 하는 의문을 가질 수 있다.
하지만 딥러닝 모델을 학습시킨다는 것은 모델이 가지고 있는 뉴런들의 가중치를 조정한다는 의미이고,
개발자는 모델 구성과 가중치만 저장 해놓으면 필요할 때 불러와서 사용하면 된다.

1. 간단한 모델 살펴보기
2. 실무에서의 딥러닝 시스템
3. 학습된 모델 저장하기
4. 모델 아키텍처 보기
5. 학습된 모델 불러오기





### 간단한 모델 살펴보기

이전까지 사용했던 mnist에서 손글씨 숫자 데이터셋을 가져와서 예측하는 예제를 사용할 것이다.

```python
# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터셋 전처리
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# 원핫인코딩 (one-hot encoding) 처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 훈련셋과 검증셋 분리
x_val = x_train[:42000] # 훈련셋의 30%를 검증셋으로 사용
x_train = x_train[42000:]
y_val = y_train[:42000] # 훈련셋의 30%를 검증셋으로 사용
y_train = y_train[42000:]

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))

# 5. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('')
print('loss_and_accuracy : ' + str(loss_and_metrics))

# 6. 모델 사용하기
xhat_idx = np.random.choice(x_test.shape[0], 5)
xhat = x_test[xhat_idx]
yhat = model.predict_classes(xhat)

for i in range(5):
    print('True : ' + str(argmax(y_test[xhat_idx[i]])) + 
          ', Predict : ' + str(yhat[i]))
```

이 코드는 **딥러닝 모델 학습**(5까지)과 **딥러닝 모델 판정**(6)이 모두 포함되어 있다. 
이 두 가지를 분리하여 우리가 원하는 저장과 불러오기를 구현해보자.



### 학습된 모델 저장하기

학습된 모델을 저장한다는 말은 '모델 아키텍처'와 '모델 가중치'를 저장한다는 말이다.
케라스에서는 `save()` 함수 하나로 둘 모두를 'h5'파일 형식으로 저장할 수 있다.

```python
from keras.models import load_model

model.save('mnist_mlp_model.h5')
```

```python
# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax

# 1. 데이터셋 생성하기

# 훈련셋과 시험셋 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터셋 전처리
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# 원핫인코딩 (one-hot encoding) 처리
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 훈련셋과 검증셋 분리
x_val = x_train[:42000] # 훈련셋의 30%를 검증셋으로 사용
x_train = x_train[42000:]
y_val = y_train[:42000] # 훈련셋의 30%를 검증셋으로 사용
y_train = y_train[42000:]

# 2. 모델 구성하기
model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 3. 모델 학습과정 설정하기
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 4. 모델 학습시키기
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val))

# 5. 모델 평가하기
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('')
print('loss_and_metrics : ' + str(loss_and_metrics))

# 6. 모델 저장하기
from keras.models import load_model
model.save('mnist_mlp_model.h5')
```

'mnist_mlp_model.h5'라는 파일이 작업 디렉토리에 생성되었는지 확인해보자. 저장된 파일에는 다음의 정보가 담겨있다.

- 나중에 모델을 재구성하기 위한 모델의 구성 정보
- 모델을 구성하는 각 뉴런들의 가중치
- 손실함수, 최적화기 등의 학습 설정
- 재학습을 할 수 있도록 마지막 학습 상태



### 모델 아키텍처 보기

model 객체를 생성한 뒤라면 언제든지 `model_to_dot()` 함수를 통해 모델 아키텍처를 블록 형태로 가시화 시킬 수 있다. 

```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

%matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
```



![svg](http://tykimos.github.io/warehouse/2017-6-10-Model_Load_Save_2.svg) 



### 학습된 모델 불러오기

'mnist_mlp_model.h5'에는 모델 아키텍처와 학습된 모델 가중치가 저장되어 있다.

- 모델을 불러오는 함수를 이용해 앞서 저장한 모델 파일로 부터 모델을 재형성한다.

- 실제 데이터로 모델을 사용한다. 이 때 주로 사용되는 함수가 `predict()`, `predict_classes()` 이다.

  `predict_classes()`  함수는 순차 기반의 분류 모델을 사용할 경우 좀 더 편리할 수 있도록 
  가장 확률이 높은 클래스 인덱스를 알려준다.



```python
# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax

# 1. 실무에 사용할 데이터 준비하기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_test = np_utils.to_categorical(y_test)
xhat_idx = np.random.choice(x_test.shape[0], 5)
xhat = x_test[xhat_idx]

# 2. 모델 불러오기
from keras.models import load_model
model = load_model('mnist_mlp_model.h5')

# 3. 모델 사용하기
yhat = model.predict_classes(xhat)

for i in range(5):
    print('True : ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))
```

```python
True : 1, Predict : 1
True : 2, Predict : 2
True : 8, Predict : 8
True : 9, Predict : 9
True : 4, Predict : 4
```

파일으로부터 모델 아키텍처와 모델 가중치를 재구성한 모델의 결과가 잘 나오는 것을 확인할 수 있다.













