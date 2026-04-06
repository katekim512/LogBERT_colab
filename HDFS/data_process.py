import sys

sys.path.append("../")

import json
import os
import re

import pandas as pd

from logdeep.dataset.semantic import attach_semantic_ids, prepare_semantic_ids, write_sequence_files


input_dir = os.path.expanduser("/content/.dataset/hdfs/")
output_dir = "../output/hdfs/"
log_file = "HDFS.log"
log_sequence_file = output_dir + "hdfs_sequence.csv"

HDFS_LOG_PATTERN = re.compile(
    r"^(?P<Date>\d{6})\s+(?P<Time>\d{6})\s+(?P<Pid>\d+)\s+"
    r"(?P<Level>\w+)\s+(?P<Component>[^:]+):\s+(?P<Content>.*)$"
)


def load_hdfs_raw_logs(log_path):
    rows = []
    with open(log_path, encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle):
            match = HDFS_LOG_PATTERN.match(line.rstrip("\n"))
            if not match:
                continue

            content = match.group("Content")
            block_ids = sorted(set(re.findall(r"(blk_-?\d+)", content)))
            if not block_ids:
                continue

            timestamp = pd.to_datetime(
                f"{match.group('Date')} {match.group('Time')}",
                format="%y%m%d %H%M%S",
                errors="coerce",
            )
            if pd.isna(timestamp):
                continue

            rows.append(
                {
                    "line_number": line_number,
                    "datetime": timestamp,
                    "Content": content,
                    "BlockIds": block_ids,
                }
            )

    return pd.DataFrame(rows)


def generate_sequences(df, label_path):
    label_df = pd.read_csv(label_path)
    blk_label_dict = {
        row["BlockId"]: 1 if row["Label"] == "Anomaly" else 0
        for _, row in label_df.iterrows()
    }

    data_dict = {}
    for _, row in df.iterrows():
        for block_id in row["BlockIds"]:
            data_dict.setdefault(block_id, []).append(row["semantic_id"])

    seq_df = pd.DataFrame(list(data_dict.items()), columns=["BlockId", "EventSequence"])
    seq_df["Label"] = seq_df["BlockId"].map(lambda block_id: blk_label_dict.get(block_id, 0))
    seq_df.to_csv(log_sequence_file, index=False)
    return seq_df


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    df = load_hdfs_raw_logs(os.path.join(input_dir, log_file))
    prepare_semantic_ids(df["Content"].tolist(), output_dir=output_dir)
    with open(os.path.join(output_dir, "semantic_id_lookup.json"), "r") as handle:
        sid_lookup = json.load(handle)["hash_to_sid"]
    df = attach_semantic_ids(df, "Content", sid_lookup)

    seq_df = generate_sequences(df, os.path.join(input_dir, "anomaly_label.csv"))

    normal_seq = seq_df[seq_df["Label"] == 0]["EventSequence"]
    normal_seq = normal_seq.sample(frac=1, random_state=20)
    abnormal_seq = seq_df[seq_df["Label"] == 1]["EventSequence"]

    normal_len = len(normal_seq)
    abnormal_len = len(abnormal_seq)
    train_len = min(4855, normal_len)
    print(f"normal size {normal_len}, abnormal size {abnormal_len}, training size {train_len}")

    write_sequence_files(output_dir + "train", normal_seq.iloc[:train_len].tolist())
    write_sequence_files(output_dir + "test_normal", normal_seq.iloc[train_len:].tolist())
    write_sequence_files(output_dir + "test_abnormal", abnormal_seq.tolist())
    print("generate train test data done")
