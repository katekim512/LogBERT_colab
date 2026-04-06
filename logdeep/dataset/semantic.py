import hashlib
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


SPECIAL_TOKENS = {"<pad>", "<unk>", "<eos>", "<sos>", "<mask>"}


def content_hash(text):
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def infer_delta_seconds(timestamps):
    values = pd.Series(timestamps)
    delta = values.diff().dt.total_seconds().fillna(0.0)
    return delta.astype(float)


def rle_count_seq(seq):
    if len(seq) == 0:
        return []

    out = []
    i = 0
    n = len(seq)
    while i < n:
        value = seq[i]
        j = i + 1
        while j < n and seq[j] == value:
            j += 1
        run_len = j - i
        out.extend([run_len] * run_len)
        i = j
    return out


def write_sequence_files(token_path, sequences):
    freq_path = token_path + "_freq"
    with open(token_path, "w") as token_file, open(freq_path, "w") as freq_file:
        for seq in sequences:
            token_file.write(" ".join(str(token) for token in seq))
            token_file.write("\n")

            freq_seq = rle_count_seq(seq)
            freq_file.write(" ".join(str(token) for token in freq_seq))
            freq_file.write("\n")


def _encode_unique_logs(unique_logs, model_name, batch_size):
    print(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    embeddings = model.encode(
        unique_logs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings.astype(np.float32)


def _residual_quantize(embeddings, num_codebooks, codebook_size, random_state):
    num_samples, dim = embeddings.shape
    residual = embeddings.copy()
    reconstruction = np.zeros_like(embeddings)
    assignments = []
    codebooks = []

    for stage in range(num_codebooks):
        stage_size = max(1, min(codebook_size, num_samples))
        quantizer = MiniBatchKMeans(
            n_clusters=stage_size,
            random_state=random_state + stage,
            batch_size=min(4096, max(stage_size * 8, 256)),
            n_init=10,
        )
        codes = quantizer.fit_predict(residual)
        centers = quantizer.cluster_centers_.astype(np.float32)
        reconstruction += centers[codes]
        residual = embeddings - reconstruction
        assignments.append(codes.astype(np.int32))
        codebooks.append(centers)

    return np.stack(assignments, axis=1), reconstruction.astype(np.float32), codebooks


def _project_embeddings(embeddings, target_dim):
    if embeddings.shape[1] == target_dim:
        return embeddings.astype(np.float32)

    n_components = min(target_dim, embeddings.shape[0], embeddings.shape[1])
    projected = PCA(n_components=n_components, random_state=42).fit_transform(embeddings)
    if n_components < target_dim:
        pad = np.zeros((projected.shape[0], target_dim - n_components), dtype=np.float32)
        projected = np.concatenate([projected.astype(np.float32), pad], axis=1)
    return projected[:, :target_dim].astype(np.float32)


def prepare_semantic_ids(
    raw_logs,
    output_dir,
    model_name="sentence-transformers/all-mpnet-base-v2",
    num_codebooks=3,
    codebook_size=128,
    batch_size=64,
    target_dim=256,
    random_state=42,
):
    if len(raw_logs) == 0:
        raise ValueError("raw_logs must not be empty")

    os.makedirs(output_dir, exist_ok=True)

    counts = Counter(str(log) for log in raw_logs)
    unique_logs = list(counts.keys())
    unique_hashes = [content_hash(log) for log in unique_logs]

    embeddings_768 = _encode_unique_logs(unique_logs, model_name=model_name, batch_size=batch_size)
    code_matrix, reconstructed_768, codebooks = _residual_quantize(
        embeddings_768,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        random_state=random_state,
    )

    semantic_ids = ["SID_" + "_".join(str(int(code)) for code in row) for row in code_matrix]
    metadata = pd.DataFrame(
        {
            "content_hash": unique_hashes,
            "raw_log": unique_logs,
            "occurrences": [counts[log] for log in unique_logs],
            "semantic_id": semantic_ids,
        }
    )
    for idx in range(code_matrix.shape[1]):
        metadata[f"code_{idx}"] = code_matrix[:, idx]

    metadata.to_csv(os.path.join(output_dir, "semantic_log_metadata.csv"), index=False)
    np.save(os.path.join(output_dir, "semantic_log_embeddings_768.npy"), embeddings_768)
    np.savez(
        os.path.join(output_dir, "semantic_codebooks.npz"),
        **{f"codebook_{idx}": centers for idx, centers in enumerate(codebooks)},
    )

    catalog = (
        metadata.groupby("semantic_id", as_index=False)
        .agg(
            occurrences=("occurrences", "sum"),
            sample_log=("raw_log", "first"),
            content_hash=("content_hash", "first"),
            **{f"code_{idx}": (f"code_{idx}", "first") for idx in range(code_matrix.shape[1])},
        )
        .sort_values("semantic_id")
        .reset_index(drop=True)
    )

    first_seen_index = {}
    for index, sid in enumerate(semantic_ids):
        first_seen_index.setdefault(sid, index)

    sid_to_index = {}
    sid_vectors_768 = []
    for sid in catalog["semantic_id"]:
        first_index = first_seen_index[sid]
        sid_to_index[sid] = len(sid_vectors_768)
        sid_vectors_768.append(reconstructed_768[first_index])

    sid_vectors_768 = np.stack(sid_vectors_768, axis=0).astype(np.float32)
    sid_vectors_256 = _project_embeddings(sid_vectors_768, target_dim=target_dim)

    catalog.to_csv(os.path.join(output_dir, "semantic_id_catalog.csv"), index=False)
    np.save(os.path.join(output_dir, "semantic_id_vectors_768.npy"), sid_vectors_768)
    np.save(os.path.join(output_dir, "semantic_id_vectors_256.npy"), sid_vectors_256)

    with open(os.path.join(output_dir, "semantic_id_lookup.json"), "w") as lookup_file:
        json.dump(
            {
                "hash_to_sid": dict(zip(unique_hashes, semantic_ids)),
                "sid_to_index": sid_to_index,
            },
            lookup_file,
            indent=2,
        )

    return metadata, dict(zip(unique_hashes, semantic_ids))


def attach_semantic_ids(df, text_column, sid_lookup):
    frame = df.copy()
    frame["content_hash"] = frame[text_column].map(content_hash)
    frame["semantic_id"] = frame["content_hash"].map(sid_lookup)
    missing = frame["semantic_id"].isna().sum()
    if missing:
        raise KeyError(f"semantic id lookup missing {missing} rows")
    return frame


def build_sbert_weight_matrix(vocab, semantic_catalog_path, semantic_vectors_path, output_path):
    catalog = pd.read_csv(semantic_catalog_path)
    sid_vectors = np.load(semantic_vectors_path)
    sid_to_vector = {
        sid: sid_vectors[idx]
        for idx, sid in enumerate(catalog["semantic_id"].tolist())
    }

    embed_size = sid_vectors.shape[1]
    weight_matrix = np.random.uniform(-0.02, 0.02, (len(vocab), embed_size)).astype(np.float32)

    for token, idx in tqdm(vocab.stoi.items(), desc="Building semantic weights"):
        if token in SPECIAL_TOKENS:
            continue
        vector = sid_to_vector.get(token)
        if vector is not None:
            weight_matrix[idx] = vector

    np.save(output_path, weight_matrix)
    return output_path
