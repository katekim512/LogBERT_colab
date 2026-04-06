import sys

sys.path.append("../")

import gc
import json
import os

import numpy as np
import pandas as pd

from logdeep.dataset.semantic import (
    attach_semantic_ids,
    infer_delta_seconds,
    prepare_semantic_ids,
    write_sequence_files,
)
from logdeep.dataset.session import sliding_window


pd.options.mode.chained_assignment = None


def count_anomaly(log_path):
    total_size = 0
    normal_size = 0
    with open(log_path, errors="ignore") as handle:
        for line in handle:
            total_size += 1
            if line.split(maxsplit=1)[0] == "-":
                normal_size += 1
    print(f"total size {total_size}, abnormal size {total_size - normal_size}")


def sample_raw_data(data_file, output_file, sample_window_size, sample_step_size):
    sample_data = []
    labels = []
    idx = 0

    with open(data_file, "r", errors="ignore") as handle:
        for line in handle:
            labels.append(line.split(maxsplit=1)[0] != "-")
            sample_data.append(line)

            if len(labels) == sample_window_size:
                abnormal_rate = sum(np.array(labels)) / len(labels)
                print(f"{idx + 1} lines, abnormal rate {abnormal_rate}")
                break

            idx += 1
            if idx % sample_step_size == 0:
                print(f"Process {round(idx / sample_window_size * 100, 4)} % raw data", end="\r")

    with open(output_file, "w") as handle:
        handle.writelines(sample_data)

    print("Sampling done")


def _parse_tbird_timestamp(parts):
    candidates = [
        f"{parts[2]} {parts[6]}",
        " ".join(parts[4:7]),
    ]
    for candidate in candidates:
        timestamp = pd.to_datetime(candidate, errors="coerce")
        if not pd.isna(timestamp):
            return timestamp
    return pd.NaT


def load_tbird_raw_logs(log_path):
    rows = []
    with open(log_path, encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle):
            parts = line.rstrip("\n").split(maxsplit=8)
            if len(parts) < 9:
                continue

            timestamp = _parse_tbird_timestamp(parts)
            if pd.isna(timestamp):
                timestamp = pd.Timestamp("1970-01-01") + pd.to_timedelta(line_number, unit="s")

            rows.append(
                {
                    "line_number": line_number,
                    "Label": int(parts[0] != "-"),
                    "datetime": timestamp,
                    "Content": parts[8],
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    data_dir = os.path.expanduser("~/.dataset/tbird/")
    output_dir = "../output/tbird/"
    raw_log_file = "Thunderbird.log"
    sample_log_file = "Thunderbird_20M.log"
    sample_window_size = 2 * 10 ** 7
    sample_step_size = 10 ** 4
    window_name = ""

    window_size = 1
    step_size = 0.5
    train_ratio = 6000

    sample_raw_data(
        os.path.join(data_dir, raw_log_file),
        os.path.join(data_dir, sample_log_file),
        sample_window_size,
        sample_step_size,
    )

    os.makedirs(output_dir, exist_ok=True)
    df = load_tbird_raw_logs(os.path.join(data_dir, sample_log_file))
    prepare_semantic_ids(df["Content"].tolist(), output_dir=output_dir)
    with open(os.path.join(output_dir, "semantic_id_lookup.json"), "r") as handle:
        sid_lookup = json.load(handle)["hash_to_sid"]
    df = attach_semantic_ids(df, "Content", sid_lookup)

    df["timestamp"] = df["datetime"].astype("int64") // 10 ** 9
    df["deltaT"] = infer_delta_seconds(df["datetime"])

    deeplog_df = sliding_window(
        df[["timestamp", "Label", "semantic_id", "deltaT"]],
        para={"window_size": float(window_size) * 60, "step_size": float(step_size) * 60},
    )
    output_dir += window_name

    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True)
    normal_len = len(df_normal)
    train_len = int(train_ratio) if train_ratio >= 1 else int(normal_len * train_ratio)

    train = df_normal[:train_len]
    write_sequence_files(os.path.join(output_dir, "train"), train["semantic_id"].tolist())
    print(f"training size {train_len}")

    test_normal = df_normal[train_len:]
    write_sequence_files(os.path.join(output_dir, "test_normal"), test_normal["semantic_id"].tolist())
    print(f"test normal size {normal_len - train_len}")

    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    write_sequence_files(os.path.join(output_dir, "test_abnormal"), df_abnormal["semantic_id"].tolist())
    print(f"test abnormal size {len(df_abnormal)}")

    del df_normal
    del train
    del test_normal
    gc.collect()
