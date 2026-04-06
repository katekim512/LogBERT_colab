import sys

sys.path.append("../")

import gc
import os

import pandas as pd

from logdeep.dataset.semantic import write_sequence_files
from logdeep.dataset.session import sliding_window
from logparser import Drain, IPLoM, Spell


pd.options.mode.chained_assignment = None

data_dir = os.path.expanduser("/content/.dataset/bgl/")
output_dir = "../output/bgl/"
log_file = "BGL.log"


def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = "<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>"
    regex = [
        r"\s+",
        r"(0x)[0-9a-fA-F]+",
        r"\d+.\d+.\d+.\d+",
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
        )
    elif parser_type == "spell":
        parser = Spell.LogParser(
            indir=input_dir,
            outdir=output_dir,
            log_format=log_format,
            tau=0.55,
            rex=regex,
            keep_para=keep_para,
        )
    elif parser_type == "iplom":
        parser = IPLoM.LogParser(log_format=log_format, indir=input_dir, outdir=output_dir, rex=regex)
    else:
        raise ValueError(f"Unsupported parser type: {parser_type}")

    parser.parse(log_file)


if __name__ == "__main__":
    window_size = 5
    step_size = 1
    train_ratio = 0.4

    os.makedirs(output_dir, exist_ok=True)
    parse_log(data_dir, output_dir, log_file, "drain")

    df = pd.read_csv(f"{output_dir}{log_file}_structured.csv")
    df["datetime"] = pd.to_datetime(df["Time"], format="%Y-%m-%d-%H.%M.%S.%f")
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    df["timestamp"] = df["datetime"].values.astype("int64") // 10 ** 9
    df["deltaT"] = df["datetime"].diff() / pd.Timedelta(seconds=1)
    df["deltaT"] = df["deltaT"].fillna(0)
    df["EventId"] = df["EventId"].astype(str)

    deeplog_df = sliding_window(
        df[["timestamp", "Label", "EventId", "deltaT"]],
        para={"window_size": int(window_size) * 60, "step_size": int(step_size) * 60},
    )

    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True)
    normal_len = len(df_normal)
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    write_sequence_files(os.path.join(output_dir, "train"), train["EventId"].tolist())
    print(f"training size {train_len}")

    test_normal = df_normal[train_len:]
    write_sequence_files(os.path.join(output_dir, "test_normal"), test_normal["EventId"].tolist())
    print(f"test normal size {normal_len - train_len}")

    del df_normal
    del train
    del test_normal
    gc.collect()

    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    write_sequence_files(os.path.join(output_dir, "test_abnormal"), df_abnormal["EventId"].tolist())
    print(f"test abnormal size {len(df_abnormal)}")
