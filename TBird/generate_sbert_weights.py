import os
import pickle
import sys

sys.path.append(os.getcwd())
sys.path.append("/content/LogBERT_colab")

from logdeep.dataset.semantic import (
    build_parser_semantic_id_weight_matrix,
    build_template_sbert_weight_matrix,
)


def generate_weights():
    output_dir = "/content/LogBERT_colab/output/tbird/"
    vocab_path = os.path.join(output_dir, "vocab.pkl")
    template_csv_path = os.path.join(output_dir, "Thunderbird_20M.log_templates.csv")
    structured_csv_path = os.path.join(output_dir, "Thunderbird_20M.log_structured.csv")
    output_path = os.path.join(output_dir, "sbert_weights.npy")
    semantic_id_output_path = os.path.join(output_dir, "semantic_id_weights.npy")

    with open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)

    build_template_sbert_weight_matrix(vocab, template_csv_path, output_path)
    build_parser_semantic_id_weight_matrix(vocab, structured_csv_path, semantic_id_output_path)
    print(f"Successfully saved semantic weights to {output_path}")
    print(f"Successfully saved semantic ID weights to {semantic_id_output_path}")


if __name__ == "__main__":
    generate_weights()
