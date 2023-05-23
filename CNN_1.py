import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 입력 데이터 형태 변환: (샘플 수, 높이, 너비, 채널)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 입력 데이터 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# 출력 데이터 원-핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 모델 구성
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
