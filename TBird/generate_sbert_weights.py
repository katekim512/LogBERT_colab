import os
import pickle
import sys

sys.path.append(os.getcwd())
sys.path.append("/content/LogBERT_colab")

from logdeep.dataset.semantic import build_sbert_weight_matrix


def generate_weights():
    output_dir = "/content/LogBERT_colab/output/tbird/"
    vocab_path = os.path.join(output_dir, "vocab.pkl")
    semantic_catalog_path = os.path.join(output_dir, "semantic_id_catalog.csv")
    semantic_vectors_path = os.path.join(output_dir, "semantic_id_vectors_256.npy")
    output_path = os.path.join(output_dir, "sbert_weights.npy")

    with open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)

    build_sbert_weight_matrix(vocab, semantic_catalog_path, semantic_vectors_path, output_path)
    print(f"Successfully saved semantic weights to {output_path}")


if __name__ == "__main__":
    generate_weights()
