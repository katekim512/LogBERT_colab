import argparse
import os
import pickle
import sys

sys.path.append(os.getcwd())
sys.path.append("/content/LogBERT_colab")

from ablation import add_ablation_argument, is_semantic_id_like, is_semparser_like
from logdeep.dataset.semantic import (
    build_dense_sbert_weight_matrix,
    build_parser_semantic_id_weight_matrix,
    build_template_sbert_weight_matrix,
)


def generate_weights_bgl(ablation):
    output_dir = "/content/LogBERT_colab/output/bgl/"
    vocab_path = os.path.join(output_dir, "vocab.pkl")
    template_csv_path = os.path.join(output_dir, "BGL.log_templates.csv")
    structured_csv_path = os.path.join(output_dir, "BGL.log_structured.csv")
    output_path = os.path.join(output_dir, "sbert_weights.npy")
    semantic_id_output_path = os.path.join(output_dir, "semantic_id_weights.npy")

    with open(vocab_path, "rb") as handle:
        vocab = pickle.load(handle)

    if is_semantic_id_like(ablation):
        build_template_sbert_weight_matrix(vocab, template_csv_path, output_path)
        build_parser_semantic_id_weight_matrix(vocab, structured_csv_path, semantic_id_output_path)
        print(f"Saved: {output_path}")
        print(f"Saved: {semantic_id_output_path}")
    elif is_semparser_like(ablation):
        build_dense_sbert_weight_matrix(vocab, template_csv_path, output_path)
        print(f"Saved: {output_path}")
    else:
        print("main ablation does not require SBERT weight generation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_ablation_argument(parser)
    args = parser.parse_args()
    generate_weights_bgl(args.ablation)
