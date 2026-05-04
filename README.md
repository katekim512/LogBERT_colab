# Semantic and Frequency-Aware Hybrid Log Embedding for Anomaly Detection

### 의미·빈도 정보를 반영한 로그 복합 임베딩 기반 이상 탐지 기법

로그 파싱 과정에서 발생하는 정보 손실 문제를 해결하고, 로그의 질적 의미와 양적 특성을 통합적으로 포착하기 위한 다중 임베딩 융합(Multi-Embedding Fusion) 기반 이상 탐지 프레임워크를 제안


<img width="4317" height="1700" alt="image" src="https://github.com/user-attachments/assets/5982becd-1ddc-4fd8-b270-5580b7d0c05d" />


### 🚀 핵심 메커니즘 (Key Mechanisms)
본 연구는 로그 데이터를 다각도에서 분석하기 위해 세 가지 독립적인 임베딩을 설계하고 이를 융합
- **Event Template Embedding**
  - 파싱된 이벤트 템플릿을 Sentence-BERT로 임베딩하여 템플릿 간 의미적 관계를 반영

- **Frequency Embedding**
  - 연속적으로 반복되는 동일 이벤트의 발생 횟수를 Run-Length Encoding 방식으로 추출하고, 이를 MLP를 통해 벡터화

- **Semantic ID Embedding**
  - 원시 로그 문장을 Sentence-BERT로 인코딩한 뒤 Residual Quantization을 적용하여 의미 기반의 이산 ID를 생성

최종적으로 각 표현을 LogBERT 입력에 결합하여 로그 시퀀스의 의미적 정보와 반복 패턴을 함께 반영

---

### ✔️ Project Structure
```bash
.
├── data_process.py              # 로그 전처리 및 ablation별 입력 데이터 생성
├── generate_sbert_weights.py    # SBERT 기반 임베딩 weight 생성
├── logbert.py                   # LogBERT 학습 및 예측 실행
├── data/                        # 원본 로그 데이터
├── output/                      # 전처리 결과 및 모델 결과 저장
└── README.md
```

---

### 🛠 실행 방법 (Usage Guide)
본 코드는 연구의 독창성을 검증하기 위한 Ablation Study 모드를 지원하며, 각 모듈을 독립적으로 실행 가능

다음 네 가지 설정으로 실행할 수 있음

| Option       | Description                                      |
|-------------|--------------------------------------------------|
| main        | 기본 LogBERT 설정                                |
| semparser   | Event Template Semantic Embedding 적용           |
| Freq        | Frequency Embedding 적용                         |
| semantic_id | Semantic ID Embedding 적용                       |


#### How to Run

전체 실행 과정은 다음 순서로 진행
```
1. 데이터 전처리
2. SBERT weight 생성
3. 모델 학습
4. 이상 탐지 예측
```
1. Data Processing :
```python data_process.py --ablation (mode)```

2. Generate SBERT Weights : 
```python generate_sbert_weights.py --ablation (mode)```


3. Train : 
```python logbert.py train --ablation (mode)```


4. Predict : 
```python logbert.py predict --ablation (mode)```

**=> (mode)에 해당 option 명을 입력**

---

### 📊 실험 데이터셋 (Datasets)

본 연구는 로그 이상 탐지 분야의 대표적인 벤치마크 데이터셋을 사용하여 검증되었습니다:

- **HDFS**: 분산 파일 시스템 로그
- **BGL**: 블루진 슈퍼컴퓨터 로그
- **TBird**: 썬더버드 슈퍼컴퓨터 로그

---

### 📚 참고 문헌 (References)

- [1] Haixuan Guo et al., "LogBERT: Log Anomaly Detection via BERT", 2021.
- [2] Min Du et al., "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning", 2017.
- [3] Reimers and Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", 2019.
