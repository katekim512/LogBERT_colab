import sys
sys.path.append('../')

import gc
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from ablation import add_ablation_argument, is_semantic_id_like, is_semparser_like
from logdeep.dataset.session import sliding_window
from logparser import Drain, Spell


pd.options.mode.chained_assignment = None


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


def deeplog_file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')


def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
    regex = [
        r'(0x)[0-9a-fA-F]+',
        r'\d+\.\d+\.\d+\.\d+',
        r'(?<=Warning: we failed to resolve data source name )[\w\s]+',
        r'\d+'
    ]
    keep_para = False
    if parser_type == "drain":
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=3, st=0.3, rex=regex, keep_para=keep_para, maxChild=1000)
    elif parser_type == "spell":
        parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=0.35, rex=regex, keep_para=keep_para)
    else:
        raise ValueError(f"Unsupported parser type: {parser_type}")
    parser.parse(log_file)


def sample_raw_data(data_file, output_file, sample_window_size, sample_step_size):
    sample_data = []
    labels = []
    idx = 0

    with open(data_file, 'r', errors='ignore') as f:
        for line in f:
            labels.append(line.split()[0] != '-')
            sample_data.append(line)

            if len(labels) == sample_window_size:
                abnormal_rate = sum(np.array(labels)) / len(labels)
                print(f"{idx + 1} lines, abnormal rate {abnormal_rate}")
                break

            idx += 1
            if idx % sample_step_size == 0:
                print(f"Process {round(idx/sample_window_size * 100,4)} % raw data", end='\r')

    with open(output_file, "w") as f:
        f.writelines(sample_data)

    print("Sampling done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_ablation_argument(parser)
    args = parser.parse_args()

    data_dir = os.path.expanduser("~/.dataset/tbird/")
    output_dir = "../output/tbird/"
    raw_log_file = "Thunderbird.log"
    sample_log_file = "Thunderbird_20M.log"
    sample_window_size = 2 * 10 ** 7
    sample_step_size = 10 ** 4
    window_name = ''
    log_file = sample_log_file

    parser_type = 'drain'
    window_size = 1
    step_size = 0.5
    train_ratio = 6000

    sample_raw_data(data_dir + raw_log_file, data_dir + sample_log_file, sample_window_size, sample_step_size)
    parse_log(data_dir, output_dir, log_file, parser_type)

    df = pd.read_csv(f'{output_dir}{log_file}_structured.csv')
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))

    time_format = '%Y.%m.%d %H:%M:%S' if is_semparser_like(args.ablation) else '%Y-%m-%d %H:%M:%S'
    df['datetime'] = pd.to_datetime(df["Date"] + " " + df['Time'], format=time_format)
    df['timestamp'] = df["datetime"].values.astype("int64") // 10 ** 9
    df['deltaT'] = df['datetime'].diff() / pd.Timedelta(seconds=1)
    df['deltaT'] = df['deltaT'].fillna(0)
    if is_semantic_id_like(args.ablation):
        df["EventId"] = df["EventId"].astype(str)

    deeplog_df = sliding_window(
        df[["timestamp", "Label", "EventId", "deltaT"]],
        para={"window_size": float(window_size) * 60, "step_size": float(step_size) * 60}
    )
    output_dir += window_name

    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True)
    normal_len = len(df_normal)
    train_len = int(train_ratio) if train_ratio >= 1 else int(normal_len * train_ratio)

    train = df_normal[:train_len]
    test_normal = df_normal[train_len:]

    if is_semantic_id_like(args.ablation):
        write_sequence_files(os.path.join(output_dir, 'train'), train["EventId"].tolist())
        write_sequence_files(os.path.join(output_dir, 'test_normal'), test_normal["EventId"].tolist())
    else:
        deeplog_file_generator(os.path.join(output_dir, 'train'), train, ["EventId"])
        deeplog_file_generator(os.path.join(output_dir, 'test_normal'), test_normal, ["EventId"])

    print("training size {}".format(train_len))
    print("test normal size {}".format(normal_len - train_len))

    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    if is_semantic_id_like(args.ablation):
        write_sequence_files(os.path.join(output_dir, 'test_abnormal'), df_abnormal["EventId"].tolist())
    else:
        deeplog_file_generator(os.path.join(output_dir, 'test_abnormal'), df_abnormal, ["EventId"])
    print('test abnormal size {}'.format(len(df_abnormal)))

    if is_semantic_id_like(args.ablation):
        del df_normal
        del train
        del test_normal
        gc.collect()
