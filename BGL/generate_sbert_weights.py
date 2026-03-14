import sys
import os
import pickle
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm

current_dir = os.getcwd()
sys.path.append(current_dir) 
sys.path.append('/content/LogBERT_colab')

def generate_weights_bgl():
    vocab_path = '/content/LogBERT_colab/output/bgl/vocab.pkl'
    template_csv_path = '/content/LogBERT_colab/output/bgl/BGL.log_templates.csv'
    output_dir = '/content/LogBERT_colab/output/bgl/'
    output_path = os.path.join(output_dir, 'sbert_weights.npy')

    if not os.path.exists(vocab_path):
        print(f"Error: {vocab_path} 파일을 찾을 수 없습니다.")
        return

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    print("Loading SBERT model...")
    word_embedding_model = models.Transformer('sentence-transformers/all-mpnet-base-v2')
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_layer1 = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(), # 768
        out_features=512, 
        activation_function=torch.nn.GELU() 
    )

    # 두 번째 층: 최종적으로 LogBERT의 입력 차원인 256으로 맞춤
    dense_layer2 = models.Dense(
        in_features=512, 
        out_features=256, 
        activation_function=torch.nn.Identity() # 마지막은 선형적으로 유지
    )

    # 4) 모든 모듈을 합쳐서 하나의 모델로 생성
    # dense_model 대신 dense_layer1, dense_layer2를 순서대로 넣습니다.
    sbert_model = SentenceTransformer(modules=[
        word_embedding_model, 
        pooling_model, 
        dense_layer1, 
        dense_layer2
    ])
    
    df_temp = pd.read_csv(template_csv_path)
    id_to_template = dict(zip(df_temp['EventId'], df_temp['EventTemplate']))

    sbert_dim = 256
    weight_matrix = np.zeros((len(vocab), sbert_dim))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert_model.to(device)
    
    for word, idx in tqdm(vocab.stoi.items()):
        if word in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
            weight_matrix[idx] = np.random.uniform(-0.02, 0.02, sbert_dim)
        elif word in id_to_template:
            template_text = str(id_to_template[word])
            with torch.no_grad():
                vector = sbert_model.encode(template_text, show_progress_bar=False)
            weight_matrix[idx] = vector
        else:
            weight_matrix[idx] = np.random.uniform(-0.02, 0.02, sbert_dim)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_path, weight_matrix)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    generate_weights_bgl()
