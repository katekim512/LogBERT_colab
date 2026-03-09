import sys
import os

# 1. 파이썬이 bert_pytorch 모듈을 찾을 수 있도록 경로 등록
# Colab 환경의 프로젝트 루트 경로를 가장 먼저 추가합니다.
current_dir = os.getcwd() # 현재 실행 경로
sys.path.append(current_dir) 
sys.path.append('/content/LogBERT_colab')

import pickle
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm
import os

def generate_weights():
    # 1. 경로 설정 
    vocab_path = '/content/LogBERT_colab/output/hdfs/vocab.pkl'
    template_csv_path = '/content/LogBERT_colab/output/hdfs/HDFS.log_templates.csv'
    output_dir = '/content/LogBERT_colab/output/hdfs/'
    output_path = os.path.join(output_dir, 'sbert_weights.npy')

    # 2. Vocab 로드 ( stoi: String to Index )
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # 3. SBERT 모델 구성 (순서가 중요합니다!)
    print("Loading SBERT model and adding Dense layer for 256 dim...")
    
    # 1) Transformer 레이어 정의 (384차원 베이스)
    word_embedding_model = models.Transformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 2) Pooling 레이어 정의 (문장 벡터화 방식 결정)
    # 이 줄이 dense_model보다 위에 있어야 합니다!
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    
    # 3) Dense 레이어 정의 (384 -> 256 차원 축소)
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(), 
        out_features=256, 
        activation_function=torch.nn.Identity()
    )

    # 4) 모든 모듈을 합쳐서 하나의 모델로 생성
    sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    
    # 4. 템플릿 데이터 로드
    df_temp = pd.read_csv(template_csv_path)
    id_to_template = dict(zip(df_temp['EventId'], df_temp['EventTemplate']))

    # 5. 가중치 행렬 초기화 [Vocab Size, 384]
    sbert_dim = 256
    weight_matrix = np.zeros((len(vocab), sbert_dim))
    
    print(f"Generating embeddings for {len(vocab)} tokens...")

    # SBERT 연산 가속을 위해 모델을 GPU로 보냄 (가능할 경우)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert_model.to(device)
    
    for word, idx in tqdm(vocab.stoi.items()):
        # 특수 토큰 처리
        if word in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
            weight_matrix[idx] = np.random.uniform(-0.02, 0.02, sbert_dim)
        
        # 실제 로그 ID (Hash 값) 인 경우
        elif word in id_to_template:
            template_text = id_to_template[word]
            # SBERT로 문장을 벡터화
            with torch.no_grad():
                vector = sbert_model.encode(template_text, show_progress_bar=False)
            weight_matrix[idx] = vector
            
        else:
            weight_matrix[idx] = np.random.uniform(-0.02, 0.02, sbert_dim)

    # 6. 저장
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_path, weight_matrix)
    print(f"Successfully saved semantic weights to {output_path}")
    print(f"Matrix Shape: {weight_matrix.shape}")

if __name__ == "__main__":
    generate_weights()