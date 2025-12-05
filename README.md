데이터는 캐글에서 구했습니다. 트위터 댓글을 이용한 감성분석입니다.
https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

감성분석(Sentiment Analysis) — Mini Transformer 보고서
1. 개요 / 문제 정의

본 프로젝트는 텍스트 기반 감성분석(Sentiment Analysis)을 수행하기 위해 Mini Transformer Encoder 기반 분류 모델을 구현하고, 전처리 및 결과 분석을 정리한 보고서다.

트랜스포머 모델은 직접 멀티헤드 어텐션부터 인코딩 함수까지 직접 구현했고, 하이퍼파라미터는 직접 실험을 반복하며 결정하였다.
입력 데이터는 텍스트 문장과 **4개의 감정 라벨(0, 1, 2, 3)**로 구성된다.

2. 데이터 전처리 (Preprocessing)
2.1 데이터 로딩

CSV 파일에서 text, label 컬럼 불러오기

결측 텍스트 제거

2.2 텍스트 정규화

소문자 변환

특수문자 제거 (한글/영문/숫자/공백 제외)

2.3 라벨 인코딩

감정 라벨: 0, 1, 2, 3 (정수형)

2.4 토큰화

HuggingFace AutoTokenizer 활용

문장을 자동으로 토큰 단위로 분리

토큰 → 정수 ID 변환

패딩, 마스크, 특수 토큰([CLS], [SEP]) 자동 처리

3. 모델 구성 (Mini Transformer)

Mini Transformer는 Transformer Encoder를 경량화하여 텍스트 분류에 최적화한 구조이다.

구성 요소:

Embedding Layer

Positional Encoding (sin/cos)

N개의 Encoder Layer

Multi-Head Self-Attention

Feed Forward

Residual Connection

Layer Normalization

Dense + ReLU

Dropout

Output Dense (Softmax)

Self-Attention을 통해 문장 내 중요 단어에 집중하며 RNN보다 병렬처리가 가능하다.

4. 학습 환경 및 하이퍼파라미터

Optimizer: Adam

Learning Rate: 1e-4

Epochs: 30

Batch Size: 32

Loss: CategoricalCrossentropy

하이퍼파라미터 요약:

num_layers: 2

Encoder 블록 수

d_model: 128

임베딩 차원, hidden size

num_heads: 4

멀티헤드 어텐션의 head 개수

dff: 256

Feed Forward 내부 차원

input_vocab_size: 30552

토크나이저 vocabulary 크기

maximum_position_encoding: 128

최대 입력 길이

num_classes: 4

감정 클래스 수

dropout rate: 0.1

과적합 방지용

5. 학습 결과
📌 정량 성능

Precision: 0.98096

Recall: 0.98021

F1-score: 0.98055

Classification Report
precision    recall   f1-score   support

0   0.99      0.98      0.98     22542
1   0.98      0.98      0.98     18318
2   0.97      0.98      0.98     20832
3   0.99      0.98      0.98     12990

accuracy                           0.98     74682
macro avg       0.98      0.98      0.98
weighted avg    0.98      0.98      0.98


➡ **전체 정확도 98%**로 매우 우수함
➡ 클래스별 성능도 고르게 0.97 이상

6. 학습 곡선 분석
6.1 Loss

Training loss: 꾸준히 하강

Validation loss: 20~25 epoch 이후 수렴
<img width="451" height="339" alt="image" src="https://github.com/user-attachments/assets/8c6fdb55-195d-43ec-9d7b-100ccc855652" />

6.2 Accuracy

Training accuracy: 초기 빠르게 증가

Validation accuracy: 약 97~98%에서 안정적으로 수렴
<img width="439" height="327" alt="image" src="https://github.com/user-attachments/assets/1fdc1c93-63f4-4a76-a20b-017b53add432" />

💡 99% 성능도 가능성이 있음

7. 클래스별 성능 및 오분류 분석

혼동행렬 결과:

모든 클래스에서 높은 정분류율

클래스 간 불균형 영향 거의 없음

Class 0,1,2는 오분류율 1~2%

Self-Attention이 문맥-단어 관계를 효과적으로 학습한 결과
<img width="563" height="436" alt="image" src="https://github.com/user-attachments/assets/f02e81a9-ff19-4efa-91f0-b7d8aef6eca8" />

8. 결론 및 향후 개선 방향

본 모델은 Transformer Encoder 기반이지만 **BERT처럼 사전학습(MLM, NSP)**을 수행하지 않았다는 점에서 다르다.

BERT: 언어 패턴을 대규모 데이터로 미리 학습

본 모델: 랜덤 초기화된 가중치를 과제 데이터로만 학습

그럼에도 불구하고 좋은 성능을 보인 이유:

Self-Attention이 긴 문장도 정보 손실 없이 처리

병렬처리로 효율적 학습

Multi-head가 다양한 문맥 패턴을 포착

[CLS] 기반 문장 임베딩이 안정적 분류에 도움

→ 따라서 LSTM 대비 훨씬 높은 정확도(F1 0.98) 달성.

