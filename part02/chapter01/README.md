# Part 02. 딥러닝 개념잡기



## Chapter 01. 데이터셋 이야기

딥러닝 모델을 학습시키려면 데이터셋이 필요하다. 풀고자 또는 만들고자 하는 모델에 따라 데이터셋 설계도 달라진다.
이 장에선 데이터셋을 어떻게 구성하고 모델을 어떻게 검증할 지 알아본다.





### 훈련셋, 검증셋, 시험셋

수능 볼 학생이 3명이 있다고 가정하고, 이 3명 중 누가 수능을 가장 잘 볼지 알아 맞힌다 하자.


- 모의고사 5회분 : 훈련셋
- 작년 수능 문제 : 시험셋
- 학생 3명 : 모델 3개
- 올해 수능 문제 : 실제 데이터 (아직 보지 못한 데이터)

![image](https://user-images.githubusercontent.com/43429667/79685929-5fe84c80-8277-11ea-93e7-3b71cadff6d3.png)

- **학습**: 문제와 해답지를 준 후 문제를 푼 뒤 정답과 맞추어 보기.
- **평가**: 문제만 주고 풀게한 뒤 체점. 학습이 일어나지 않는다.



수능을 누가 가장 잘 볼지 여러 경우에 따라 살펴보자.



#### 경우1

올해 수능 문제로 시험을 쳐서 가장 점수가 높은 학생을 고르는게 가장 쉬운 방법일 것이다.
하지만 올해 수능 문제를 수능 전에 알아낼 수는 없다. 이것을 볼 수 없는 데이터<sup>unseen data</sup>라고 한다.

![image](https://user-images.githubusercontent.com/43429667/79686070-6dea9d00-8278-11ea-8093-c2379aa007b6.png)



#### 경우2

모의고사 5회분을 학습한 뒤 작년 수능 문제를 시험셋으로 사용?

![image](https://user-images.githubusercontent.com/43429667/79686167-447e4100-8279-11ea-9cc9-d5a7b90ab94b.png)



#### 경우3

학생들이 스스로 학습 상태를 확인하고 학습 방법을 바꾸거나 중단할 수 있다면 더 알맞은 학습을 할 수 있을 것이다.
이를 위해 **검증셋**이 필요하다. 검증셋이  있다면 스스로 평가하면서 적절한 학습방법을 찾을 수 있다.

> 모의고사 1~4회를 훈련셋, 5회를 검증셋으로 두어 학습할 때 사용하지 않는다.

![image](https://user-images.githubusercontent.com/43429667/79719483-a1313880-8319-11ea-80b7-6706b2b6dea0.png)

이 방식은 두 가지 효과를 얻을 수 있다.

1. 학습 방법을 바꾼 후 훈련셋으로 학습을 해보고 검증셋으로 평가해볼 수 있다.
   검증셋으로 가장 높은 평가를 받은 학습 방법이 최적의 학습 방법이라고 생각하면 된다.

   이러한 학습 방법을 결정하는 파라미터를 **하이퍼파라미터**<sup>hyperparameter</sup>라 하고
   최적의 학습 방법을 찾아가는 것을 하이퍼파라미터 튜닝이라 한다.

   > 모델링할 때 사용자가 직접 세팅해주는 모든 값이 하이퍼파라미터이다.



2. 얼마정도의 반복 학습이 좋을 지를 정하기 위해서 검증셋을 사용할 수 있다.
   초기에는 에포크<sup>epochs</sup>가 증가될수록 검증셋의 평가 결과도 좋아진다.

   <img src="https://user-images.githubusercontent.com/43429667/79721185-ed31ac80-831c-11ea-9262-0e1a0be7fc48.png" alt="image" style="zoom:50%;" /> 

   > 세로: 문항 수, 가로: 반복 횟수

   위의 상태는 아직 학습이 덜 된 상태, 즉 학습을 더 하면 성능이 높아질 가능성이 있는 상태로 **언더피팅**<sup>underfitting</sup>이라 한다.
   에포크를 증가시키다보면 더 이상 검증셋의 평가가 높아지는게 아니라 오버피팅되어 오히려 틀린 개수가 많아진다.
   이 시점이 적정 반복 횟수로 보고 학습을 중단한다. 이를 **조기종료**<sup>early stopping</sup>이라 한다.

   ![image](https://user-images.githubusercontent.com/43429667/79722898-d6d92000-831f-11ea-8437-6bb4a0c0667d.png)

#### 경우4

위처럼 모의고사 5회로만 검증셋을 사용할 경우 여러가지 문제가 발생할 수 있다.

- 모의고사 5회에서 출제가 되지 않는 분야가 있을 수 있다.
- 모의고사 5회가 작년 수능이나 올해 수능 문제와 많이 다를 수도 있다.
- 모의고사 5회가 모의고사 1회~4회와 난이도 및 범위가 다를 수도 있다.

이런 이유로 모의고사 5회로만 검증셋을 사용하기에는 객관적인 평가가 이루어졌다고 보기 힘들다. 
이 때 사용하는 것이 교차검증<sup>cross-validation</sup> 이다. 하는 방법은 다음과 같다.

- 모의고사 1회~4회를 학습한 뒤 모의고사 5회로 평가를 수행한다.
- 학습된 상태를 초기화한 후 다시 모의고사 1, 2, 3, 5회를 학습한 뒤 4회로 검증한다.
- 학습된 상태를 초기화한 후 다시 모의고사 1, 2, 4, 5회를 학습한 뒤 3회로 검증한다.
- 학습된 상태를 초기화한 후 다시 모의고사 1, 3, 4, 5회를 학습한 뒤 2회로 검증한다.
- 학습된 상태를 초기화한 후 다시 모의고사 2, 3, 4, 5회를 학습한 뒤 1회로 검증한다.

![image](https://user-images.githubusercontent.com/43429667/79723405-a5ad1f80-8320-11ea-89c1-61b361bbfcbf.png)



단 교차검증은 계산량이 많기 때문에 데이터수가 많지 않을 때 사용하며, 
딥러닝 모델은 대량의 데이터를 사용하므로 잘 사용되지 않는다.

### 요약

딥러닝에서는 모델 아키텍처와 함께 데이터셋은 중요한 요소이다. 
데이터셋을 훈련셋, 검증셋, 시험셋으로 나눠야하는 이유를 알아봤고, 어떤식으로 사용하는 지 알아보았다.
