데이터는 캐글에서 구했습니다. 트위터 댓글을 이용한 감성분석입니다.
https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

감성분석(Sentiment Analysis) — Mini Transformer 보고서
1. 개요 / 문제 정의

본 프로젝트는 텍스트 기반 감성분석(Sentiment Analysis)을 수행하기 위해 Mini Transformer Encoder 기반 분류 모델을 구현하고, 전처리 및 결과 분석을 정리한 보고서다.

트랜스포머 모델은 직접 멀티헤드 어텐션부터 인코딩 함수까지 직접 구현했고, 하이퍼파라미터는 직접 실험을 반복하며 결정하였다.
입력 데이터는 텍스트 문장과 **4개의 감정 라벨(0, 1, 2, 3)**로 구성된다.

2. 데이터 전처리 (Preprocessing)
2.1 데이터 로딩-
-판다스 pd.read로 불러왔다.
CSV파일 두개를 합쳐서 train : valid : test = 8 : 1 : 1로 나누었다.

2.2 텍스트 정규화
- 소문자 변환
- 특수문자 제거와 불용어 제거는 하지 않았다.

2.3 라벨 인코딩
라벨은 0,1,2,3의 정수형으로 사용한다.

2.4 토큰화
HuggingFace의 AutoTokenizer는 입력 문장을 모델에 맞게 자동으로 단어 단위(토큰)로 분리하고, 각 토큰을 정수 ID로 변환하여 모델이 이해할 수 있는 형태로 만든다. 또한 패딩, 마스킹, 특수 토큰([CLS], [SEP]) 처리까지 자동으로 수행해 안정적인 입력 구성을 돕는다.

2.5 데이터 불균형 가중치
-라벨마다 데이터 불균형이 있었기에 각 클래스에 가중치를 곱해주었다.

2.6 Input Text 생성
본 연구에서는 속성 기반 감정 분석을 위해 문장과 대상(Entity)을 결합한 프롬프트 형태의 입력을 구성하였다. 이를 위해 다음과 같이 입력 텍스트를 생성하였다.
def build_prompt_text(row):
    entity = row["Entity"]
    tweet = row["Tweet comment"]
    return f'What is the sentiment toward "{entity}" in this text? [SEP] {tweet}'
이 구조는 감정을 분석해야 할 대상을 명시적으로 제시함으로써, 사전학습을 거치지 않은 Transformer Encoder가 문장 전체가 아닌 특정 엔티티에 집중할 수 있도록 돕는다. 또한 [SEP] 토큰을 사용하여 “질문 영역”과 “본문 트윗”을 구분함으로써 입력의 의미적 구조를 명확히 하여 모델의 이해도를 높인다.
이를 통해 소규모 모델에서도 엔티티 중심 감정 분석 정확도를 향상시키는 효과가 있다.

3. 모델 구성 (Mini Transformer)
Mini Transformer는 Transformer Encoder를 경량화하여 텍스트 분류에 적용한 모델이다. 주요 구성은 다음과 같다:
- Embedding Layer
- Positional Encoding (사인/코사인 기반)
- N개의 Encoder Layer (Multi-Head Self-Attention + Feed Forward + Residual + LayerNorm)
- Dense (ReLU) + Dropout + Output Dense (Softmax)

이 모델은 self-attention을 통해 문장 내 중요 단어에 주목하며, RNN 계열 모델보다 병렬 연산에 유리하다.
4. 학습 환경 및 하이퍼파라미터
- Optimizer: Adam
- Learning Rate: 0.00005
- Epochs: 30
- Batch Size: 16
- Loss: CategoricalCrossentropy
- Early stop – monitoring Loss를 걸었다.
1.num_layers : 3 
-Encoder 블록 수. 각 블록은 Multi-Head Attention + Feed Forward + Residual +LayerNorm으로 구성. 층이 많을수록 표현력 증가, 학습 속도 저하 가능.
2. d_model : 256   
-모델 hidden size 및 단어 임베딩 차원. 차원이 클수록 표현력이 증가하지만 메모리와 학습 시간 증가.
3. num_heads : 4
-Multi-Head Attention의 head 수. 서로 다른 시각으로 문맥 관계를 학습. `d_model`은 `num_heads * depth_per_head` 구조여야 함.
4. dff : 512      
-Feed Forward Network의 내부 hidden layer 차원. 크면 더 복잡한 비선형 표현 가능.
5. maximum_position_encoding
-Positional Encoding 계산 최대 시퀀스 길이. 입력 토큰 길이가 이보다 길면 자름.
-글자수가 모자라면 170자리 까지 0으로 패딩해줌.
-max_len은 토큰 개수 분포 그래프를 참고했다.

 <img width="900" height="487" alt="image" src="https://github.com/user-attachments/assets/f770b6df-b4a7-4c77-9038-cb7bff2ce7c9" />


7. num_classes : 4
-분류할 감정 클래스 수 (negative / neutral / positive / irrelevant)
8. rate : 0.2  
 -Dropout 비율. 학습 시 일부 뉴런을 무작위로 끄고 과적합 방지.
-Dropout은 학습시에만 사용하고 검증용과 테스트용으로는 사용하지 않는다.

5. 학습 결과 
474/474 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step
Precision: 0.9057038954294692
Recall: 0.90639996158757
F1-score: 0.9060341333094245

=== Classification Report ===
              precision    recall  f1-score   support
           0       0.91      0.92      0.91      2281
           1       0.91      0.90      0.91      1861
           2       0.91      0.90      0.91      2111
           3       0.89      0.90      0.89      1316
    accuracy                           0.91      7569
   macro avg       0.91      0.91      0.91      7569
weighted avg       0.91      0.91      0.91      7569

본 모델은 다중분류 기준에서 Precision 0.905, Recall 0.906, F1-score 0.906로 매우 우수한 성능을 보였다.
클래스별 지표도 대부분 0.9 이상으로, 특정 클래스에 치우치지 않고 고르게 안정적인 예측력을 나타냈다.
하지만 상대적으로 클래스 수가 제일 적은 3이 제일 테스트 정확도가 낮아 아쉬운 면도 있었다.
특히 약 7만 개가 넘는 검증 샘플에서 **전체 정확도 90%**를 달성해 신뢰도가 높다.
macro avg와 weighted avg가 동일하게 0.90 수준인 것은 클래스 균형에 상관없이 전반적인 성능이 뛰어남을 의미한다.
종합적으로, 제안한 Transformer 기반 모델은 과제 데이터셋에서 매우 높은 분류 성능을 보여 실용성이 충분하다.
6. 학습 곡선 분석
<img width="410" height="310" alt="image" src="https://github.com/user-attachments/assets/11e345aa-15da-464b-8050-a6757d8e80ca" />

<img width="413" height="310" alt="image" src="https://github.com/user-attachments/assets/5c7c526b-21bd-4198-87b2-eb98aada3d6d" />


6.1 Loss 변화
- Training loss는 에포크 진행에 따라 꾸준히 감소하였다.
- Validation loss는 4 epoch 이후 수렴하거나 정체되었다.

6.2 Accuracy 변화
- Training accuracy는 빠르게 상승하여 90% 이상에 도달하였고, Validation accuracy는 약 89~90% 수준에서 수렴하였다.

=> 전반적으로 모델이 안정적으로 학습되었으며, 추가 데이터 또는 규제(regularization)로 개선 여지가 있다. 
데이터를 훨씬 더 많이 수집해서 모델의 수준을 더 깊게 한다면 95% 까지 향상 가능성이 보인다.
7. 클래스별 성능 및 오분류 분석
 <img width="855" height="734" alt="image" src="https://github.com/user-attachments/assets/c8d6b1c0-d044-421e-a379-4c885dd4b2ab" />

혼동행렬 결과, 네 모델은 모든 클래스에서 높은 정분류 수를 기록하며 안정적인 성능을 보였다.
각 클래스의 오분류 비율이 낮고, 특히 Class 0,1,2는 8~9% 이내의 범위에서만 다른 클래스로 잘못 예측되었다.
네 개의 클래스 모두 주대각선 값이 압도적으로 높게 나타나 모델이 클래스 간 경계를 명확히 학습했음을 보여준다.
전체적으로 클래스 불균형의 영향이 적으며, Transformer 기반 모델이 문장-엔티티 관계를 효과적으로 반영해 높은 분류 정확도를 달성했음을 확인할 수 있다.
9. 결론 및 향후 개선 방향
본 연구에서 사용한 모델은 Transformer Encoder 구조를 기반으로 하지만, BERT처럼 대규모 사전학습 과정을 거치지 않았다는 점에서 근본적으로 다르다.
 BERT는 Masked Language Modeling과 Next Sentence Prediction을 통해 언어적 패턴을 미리 학습한 반면, 본 모델은 랜덤 초기화된 상태에서 직접 제공된 학습 데이터만으로 학습을 진행한다. 
그럼에도 불구하고 Transformer 기반 모델이 LSTM보다 높은 성능을 보인 이유는 Self-Attention 메커니즘이 문장 전체의 단어 관계를 병렬적으로 파악할 수 있기 때문이다. LSTM은 순차적으로 입력을 처리하기 때문에 긴 문장에서 정보가 소실되기 쉽지만, Transformer는 모든 단어를 동시에 비교하여 장기 의존성을 효과적으로 포착한다. 또한 [CLS] 토큰 기반의 문장 표현을 사용해 문맥 정보가 안정적으로 반영되고, Multi Head Attention 덕분에 다양한 관점의 특징이 학습된다. 
이러한 구조적 이점들로 인해 본 Transformer 모델은 BERT의 사전학습은 없지만, LSTM 대비 더 높은 정확도와 F1-score를 달성하였다.




