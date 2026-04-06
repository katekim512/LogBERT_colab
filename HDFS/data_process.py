import sys

sys.path.append("../")

import ast
import os
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from logdeep.dataset.semantic import write_sequence_files
from logparser import Drain, IPLoM, Spell


input_dir = os.path.expanduser("/content/.dataset/hdfs/")
output_dir = "../output/hdfs/"
log_file = "HDFS.log"

log_structured_file = output_dir + log_file + "_structured.csv"
log_sequence_file = output_dir + "hdfs_sequence.csv"


def parse_event_sequence(row):
    if isinstance(row, list):
        return row
    if isinstance(row, str):
        return ast.literal_eval(row)
    return ast.literal_eval(str(row))


def parser(input_dir, output_dir, log_file, log_format, parser_type="drain"):
    if parser_type == "spell":
        tau = 0.5
        regex = [
            r"(/[-\w]+)+",
            r"(?<=blk_)[-\d]+",
        ]
        parser_obj = Spell.LogParser(
            indir=input_dir,
            outdir=output_dir,
            log_format=log_format,
            tau=tau,
            rex=regex,
            keep_para=False,
        )
    elif parser_type == "drain":
        regex = [
            r"(?<=blk_)[-\d]+",
            r"\d+\.\d+\.\d+\.\d+",
            r"(/[-\w]+)+",
        ]
        parser_obj = Drain.LogParser(
            log_format,
            indir=input_dir,
            outdir=output_dir,
            depth=5,
            st=0.5,
            rex=regex,
            keep_para=False,
        )
    elif parser_type == "iplom":
        regex = [
            r"(/[-\w]+)+",
            r"(?<=blk_)[-\d]+",
        ]
        parser_obj = IPLoM.LogParser(
            log_format=log_format,
            indir=input_dir,
            outdir=output_dir,
            rex=regex,
        )
    else:
        raise ValueError(f"Unsupported parser type: {parser_type}")

    parser_obj.parse(log_file)


def hdfs_sampling(log_file):
    print("Loading", log_file)
    df = pd.read_csv(
        log_file,
        engine="c",
        na_filter=False,
        memory_map=True,
        dtype={"Date": object, "Time": object},
    )

    data_dict = defaultdict(list)
    for _, row in tqdm(df.iterrows()):
        blk_id_list = re.findall(r"(blk_-?\d+)", row["Content"])
        blk_id_set = set(blk_id_list)
        for blk_id in blk_id_set:
            data_dict[blk_id].append(str(row["EventId"]))

    data_df = pd.DataFrame(list(data_dict.items()), columns=["BlockId", "EventSequence"])
    data_df.to_csv(log_sequence_file, index=None)
    print("hdfs sampling done")


def generate_train_test(hdfs_sequence_file, n=None, ratio=0.3):
    blk_label_dict = {}
    blk_label_file = os.path.join(input_dir, "anomaly_label.csv")
    blk_df = pd.read_csv(blk_label_file)
    for _, row in tqdm(blk_df.iterrows()):
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    seq = pd.read_csv(hdfs_sequence_file)
    seq["Label"] = seq["BlockId"].apply(lambda x: blk_label_dict.get(x))

    normal_seq = seq[seq["Label"] == 0]["EventSequence"]
    normal_seq = normal_seq.sample(frac=1, random_state=20)

    abnormal_seq = seq[seq["Label"] == 1]["EventSequence"]
    normal_len, abnormal_len = len(normal_seq), len(abnormal_seq)
    train_len = n if n else int(normal_len * ratio)
    print(f"normal size {normal_len}, abnormal size {abnormal_len}, training size {train_len}")

    train = normal_seq.iloc[:train_len].apply(parse_event_sequence).tolist()
    test_normal = normal_seq.iloc[train_len:].apply(parse_event_sequence).tolist()
    test_abnormal = abnormal_seq.apply(parse_event_sequence).tolist()

    write_sequence_files(output_dir + "train", train)
    write_sequence_files(output_dir + "test_normal", test_normal)
    write_sequence_files(output_dir + "test_abnormal", test_abnormal)
    print("generate train test data done")


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    log_format = "<Date> <Time> <Pid> <Level> <Component>: <Content>"
    parser(input_dir, output_dir, log_file, log_format, "drain")
    hdfs_sampling(log_structured_file)
    generate_train_test(log_sequence_file, n=4855)
