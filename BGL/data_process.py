import sys

sys.path.append("../")

import gc
import json
import os

import pandas as pd

from logdeep.dataset.semantic import (
    attach_semantic_ids,
    infer_delta_seconds,
    prepare_semantic_ids,
    write_sequence_files,
)
from logdeep.dataset.session import sliding_window


pd.options.mode.chained_assignment = None


data_dir = os.path.expanduser("/content/.dataset/bgl/")
output_dir = "../output/bgl/"
log_file = "BGL.log"


def load_bgl_raw_logs(log_path):
    rows = []
    with open(log_path, encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle):
            parts = line.rstrip("\n").split(maxsplit=9)
            if len(parts) < 10:
                continue

            timestamp = pd.to_datetime(parts[4], format="%Y-%m-%d-%H.%M.%S.%f", errors="coerce")
            if pd.isna(timestamp):
                timestamp = pd.to_datetime(parts[2] + " " + parts[4], errors="coerce")
            if pd.isna(timestamp):
                continue

            rows.append(
                {
                    "line_number": line_number,
                    "Label": int(parts[0] != "-"),
                    "datetime": timestamp,
                    "Content": parts[9],
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    window_size = 5
    step_size = 1
    train_ratio = 0.4

    os.makedirs(output_dir, exist_ok=True)

    df = load_bgl_raw_logs(os.path.join(data_dir, log_file))
    prepare_semantic_ids(df["Content"].tolist(), output_dir=output_dir)
    with open(os.path.join(output_dir, "semantic_id_lookup.json"), "r") as handle:
        sid_lookup = json.load(handle)["hash_to_sid"]
    df = attach_semantic_ids(df, "Content", sid_lookup)

    df["timestamp"] = df["datetime"].astype("int64") // 10 ** 9
    df["deltaT"] = infer_delta_seconds(df["datetime"])

    deeplog_df = sliding_window(
        df[["timestamp", "Label", "semantic_id", "deltaT"]],
        para={"window_size": int(window_size) * 60, "step_size": int(step_size) * 60},
    )

    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True)
    normal_len = len(df_normal)
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    write_sequence_files(
        os.path.join(output_dir, "train"),
        train["semantic_id"].tolist(),
    )
    print(f"training size {train_len}")

    test_normal = df_normal[train_len:]
    write_sequence_files(
        os.path.join(output_dir, "test_normal"),
        test_normal["semantic_id"].tolist(),
    )
    print(f"test normal size {normal_len - train_len}")

    del df_normal
    del train
    del test_normal
    gc.collect()

    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    write_sequence_files(
        os.path.join(output_dir, "test_abnormal"),
        df_abnormal["semantic_id"].tolist(),
    )
    print(f"test abnormal size {len(df_abnormal)}")
