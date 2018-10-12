# Copyright (c) 2016-2017, Deogtae Kim & DTWARE Inc. All rights reserved.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tf.reset_default_graph()
tf.set_random_seed(107)
 
## 데이터 수집

# ECG 훈련 데이터 및 테스트 데이터를 읽어 오기
train_ecg = pd.read_csv("ecg_discord_train.csv", header=None)
test_ecg = pd.read_csv("ecg_discord_test.csv", header=None)

## 데이터 탐색

print(train_ecg.shape)
train_ecg.head()
print(test_ecg.shape)
test_ecg.head()
 

## 학습 모델 생성: Deep Autoencoder

# 하이퍼 매개변수 설정
learning_rate = 0.0001
training_epochs = 13000
display_step = 500
examples_to_show = 10

n_input = 210
n_hidden_1 = 50 # 첫번째 층의 뉴런(특징, 속성) 갯수
n_hidden_2 = 20 # 두번째 층의 뉴런(특징, 속성) 갯수

# 텐서플로 그래프 외부 입력 홀더 생성
X = tf.placeholder("float", [None, n_input])

# 인코더 생성
EW1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
Eb1 = tf.Variable(tf.random_normal([n_hidden_1]))
EL1 = tf.nn.tanh(tf.matmul(X, EW1) + Eb1)
#EL1 = tf.nn.sigmoid(tf.matmul(X, EW1) + Eb1)

EW2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
Eb2 = tf.Variable(tf.random_normal([n_hidden_2]))
EL2 = tf.nn.tanh(tf.matmul(EL1, EW2) + Eb2)
#EL2 = tf.nn.sigmoid(tf.matmul(EL1, EW2) + Eb2)

# 디코더 생성
DW1 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]))
Db1 = tf.Variable(tf.random_normal([n_hidden_1]))
DL1 = tf.nn.tanh(tf.matmul(EL2, DW1) + Db1)
#DL1 = tf.nn.sigmoid(tf.matmul(EL2, DW1) + Db1)

DW2 = tf.Variable(tf.random_normal([n_hidden_1, n_input]))
Db2 = tf.Variable(tf.random_normal([n_input]))
DL2 = tf.matmul(DL1, DW2) + Db2

# 예측
y_pred = DL2
y_true = X

# 손실 함수 정의 및 최적화 알고리듬 설정
cost = tf.reduce_mean(tf.square(y_true-y_pred))
#train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 텐서플로 변수 초기화
init = tf.global_variables_initializer()

## 훈련

sess = tf.Session()
sess.run(init)
import time
start = time.time()
for epoch in range(training_epochs):
    _, c = sess.run([train_step, cost], feed_dict={X: train_ecg})
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
print("훈련 시간:", time.time() - start)  

## 모델 평가

test_recon = sess.run(y_pred, feed_dict={X: test_ecg})

print(test_ecg.iloc[0:5, 0:5])
print(test_recon[0:5, 0:5])
test_recon_error = ((test_recon - test_ecg) ** 2).mean(axis=1)
print(test_recon_error)
print("평균 복원 오차:", np.mean(test_recon_error))
print("정상 데이터 평균 복원 오차:", np.mean(test_recon_error[0:20]))
print("비정상 데이터 평균 복원 오차:", np.mean(test_recon_error[20:23]))

# 이상 데이터 (ouliers) (마지막 3개의 ECG 데이터)를 시각화
plt.ion()
plt.plot(test_recon_error)
plt.show()

sess.close()
