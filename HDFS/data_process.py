import sys
sys.path.append('../')

import ast
import json
import os
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from ablation import add_ablation_argument, is_semantic_id_like
from logparser import Drain, IPLoM, Spell


input_dir = os.path.expanduser('/content/.dataset/hdfs/')
output_dir = '../output/hdfs/'
log_file = "HDFS.log"

log_structured_file = output_dir + log_file + "_structured.csv"
log_templates_file = output_dir + log_file + "_templates.csv"
log_sequence_file = output_dir + "hdfs_sequence.csv"


def mapping():
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by=["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx + 1 for idx, event in enumerate(list(log_temp["EventId"]))}
    print(log_temp_dict)
    with open(output_dir + "hdfs_log_templates.json", "w") as f:
        json.dump(log_temp_dict, f)


def parse_event_sequence(row):
    if isinstance(row, list):
        return row
    if isinstance(row, str):
        return ast.literal_eval(row)
    return ast.literal_eval(str(row))


def parser(input_dir, output_dir, log_file, log_format, parser_type='drain'):
    if parser_type == 'spell':
        tau = 0.5
        regex = [
            r"(/[-\w]+)+",
            r"(?<=blk_)[-\d]+"
        ]
        parser_obj = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=False)
    elif parser_type == 'drain':
        regex = [
            r"(?<=blk_)[-\d]+",
            r'\d+\.\d+\.\d+\.\d+',
            r"(/[-\w]+)+",
        ]
        parser_obj = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=5, st=0.5, rex=regex, keep_para=False)
    elif parser_type == 'iplom':
        regex = [
            r"(/[-\w]+)+",
            r"(?<=blk_)[-\d]+"
        ]
        parser_obj = IPLoM.LogParser(log_format=log_format, indir=input_dir, outdir=output_dir, rex=regex)
    else:
        raise ValueError(f"Unsupported parser type: {parser_type}")

    parser_obj.parse(log_file)


def hdfs_sampling(log_file, ablation, window='session'):
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    print("Loading", log_file)
    df = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True, dtype={'Date': object, "Time": object})

    if not is_semantic_id_like(ablation):
        with open(output_dir + "hdfs_log_templates.json", "r") as f:
            event_num = json.load(f)
        df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))
    else:
        df["EventId"] = df["EventId"].astype(str)

    data_dict = defaultdict(list)
    for _, row in tqdm(df.iterrows()):
        blk_id_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blk_id_set = set(blk_id_list)
        for blk_id in blk_id_set:
            data_dict[blk_id].append(row["EventId"])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
    data_df.to_csv(log_sequence_file, index=None)
    print("hdfs sampling done")


def rle_count_seq(seq):
    if not seq:
        return []
    out = []
    i = 0
    n = len(seq)
    while i < n:
        v = seq[i]
        j = i + 1
        while j < n and seq[j] == v:
            j += 1
        run_len = j - i
        out.extend([run_len] * run_len)
        i = j
    return out


def df_to_file(df, file_name):
    with open(file_name, 'w') as f:
        for _, row in df.items():
            f.write(' '.join([str(ele) for ele in eval(row)]))
            f.write('\n')


def df_to_files(df, token_file_name, freq_file_name):
    with open(token_file_name, 'w') as f_tok, open(freq_file_name, 'w') as f_freq:
        for _, row in df.items():
            seq = parse_event_sequence(row)
            f_tok.write(' '.join([str(ele) for ele in seq]))
            f_tok.write('\n')
            freq_seq = rle_count_seq(seq)
            f_freq.write(' '.join([str(ele) for ele in freq_seq]))
            f_freq.write('\n')


def write_sequence_files(token_path, sequences):
    freq_path = token_path + "_freq"
    with open(token_path, "w") as token_file, open(freq_path, "w") as freq_file:
        for seq in sequences:
            token_file.write(" ".join(str(token) for token in seq))
            token_file.write("\n")
            freq_seq = rle_count_seq(seq)
            freq_file.write(" ".join(str(token) for token in freq_seq))
            freq_file.write("\n")


def generate_train_test(hdfs_sequence_file, ablation, n=None, ratio=0.3):
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
    print("normal size {0}, abnormal size {1}, training size {2}".format(normal_len, abnormal_len, train_len))

    train = normal_seq.iloc[:train_len]
    test_normal = normal_seq.iloc[train_len:]
    test_abnormal = abnormal_seq

    if is_semantic_id_like(ablation):
        write_sequence_files(output_dir + "train", train.apply(parse_event_sequence).tolist())
        write_sequence_files(output_dir + "test_normal", test_normal.apply(parse_event_sequence).tolist())
        write_sequence_files(output_dir + "test_abnormal", test_abnormal.apply(parse_event_sequence).tolist())
    elif ablation == "freq":
        df_to_files(train, output_dir + "train", output_dir + "train_freq")
        df_to_files(test_normal, output_dir + "test_normal", output_dir + "test_normal_freq")
        df_to_files(test_abnormal, output_dir + "test_abnormal", output_dir + "test_abnormal_freq")
    else:
        df_to_file(train, output_dir + "train")
        df_to_file(test_normal, output_dir + "test_normal")
        df_to_file(test_abnormal, output_dir + "test_abnormal")
    print("generate train test data done")


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    add_ablation_argument(arg_parser)
    args = arg_parser.parse_args()

    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
    parser(input_dir, output_dir, log_file, log_format, 'drain')
    if not is_semantic_id_like(args.ablation):
        mapping()
    hdfs_sampling(log_structured_file, args.ablation)
    generate_train_test(log_sequence_file, args.ablation, n=4855)
