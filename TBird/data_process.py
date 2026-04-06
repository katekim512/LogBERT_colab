import sys

sys.path.append("../")

import gc
import os

import numpy as np
import pandas as pd

from logdeep.dataset.semantic import write_sequence_files
from logdeep.dataset.session import sliding_window
from logparser import Drain, Spell


pd.options.mode.chained_assignment = None


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


def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = "<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>"
    regex = [
        r"(0x)[0-9a-fA-F]+",
        r"\d+\.\d+\.\d+\.\d+",
        r"(?<=Warning: we failed to resolve data source name )[\w\s]+",
        r"\d+",
    ]
    keep_para = False

    if parser_type == "drain":
        parser = Drain.LogParser(
            log_format,
            indir=input_dir,
            outdir=output_dir,
            depth=3,
            st=0.3,
            rex=regex,
            keep_para=keep_para,
            maxChild=1000,
        )
    elif parser_type == "spell":
        parser = Spell.LogParser(
            indir=input_dir,
            outdir=output_dir,
            log_format=log_format,
            tau=0.35,
            rex=regex,
            keep_para=keep_para,
        )
    else:
        raise ValueError(f"Unsupported parser type: {parser_type}")

    parser.parse(log_file)


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
    parse_log(data_dir, output_dir, sample_log_file, "drain")

    df = pd.read_csv(f"{output_dir}{sample_log_file}_structured.csv")
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M:%S")
    df["timestamp"] = df["datetime"].values.astype("int64") // 10 ** 9
    df["deltaT"] = df["datetime"].diff() / pd.Timedelta(seconds=1)
    df["deltaT"] = df["deltaT"].fillna(0)
    df["EventId"] = df["EventId"].astype(str)

    deeplog_df = sliding_window(
        df[["timestamp", "Label", "EventId", "deltaT"]],
        para={"window_size": float(window_size) * 60, "step_size": float(step_size) * 60},
    )
    output_dir += window_name

    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True)
    normal_len = len(df_normal)
    train_len = int(train_ratio) if train_ratio >= 1 else int(normal_len * train_ratio)

    train = df_normal[:train_len]
    write_sequence_files(os.path.join(output_dir, "train"), train["EventId"].tolist())
    print(f"training size {train_len}")

    test_normal = df_normal[train_len:]
    write_sequence_files(os.path.join(output_dir, "test_normal"), test_normal["EventId"].tolist())
    print(f"test normal size {normal_len - train_len}")

    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    write_sequence_files(os.path.join(output_dir, "test_abnormal"), df_abnormal["EventId"].tolist())
    print(f"test abnormal size {len(df_abnormal)}")

    del df_normal
    del train
    del test_normal
    gc.collect()
