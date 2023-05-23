# CNN

## MNIST 손글씨 코드

1. 컨볼루션 층에서는 ReLu 활성화 함수 사용
2. 이미지의 각 픽셀은 0~255 사이의 값으로 표현 -> 모델의 학습을 어렵게 만듦
3. 데이터 정규화를 통해 입력 데이터의 범위를 255로 나누어 0~1로 조정 -> 일반화하기 쉬움

### CNN_1
스트라이드 값은 기본적으로 (1,1)로 설정
### CNN_2
스트라이드 값을 (2,2)로 증가

## 출력값
### CNN_1
![1](https://github.com/HyunJJJUN/CNN/assets/124676369/77308b40-e11f-4546-8462-1338b4abef3b)
### CNN_2
![2](https://github.com/HyunJJJUN/CNN/assets/124676369/ab20bb53-751a-4949-b422-241c10ebed26)

(1,1) -> (2,2)
이미지를 크게 건너뛰어서 차원이 많이 축소
전의 결과보다 정확도가 감소하고 손실값은 증가
