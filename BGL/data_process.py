import sys
sys.path.append('../')

import gc
import os

import pandas as pd

from ablation import add_ablation_argument, is_semantic_id_like, is_semparser_like
from logdeep.dataset.session import sliding_window
from logparser import Drain, IPLoM, Spell


pd.options.mode.chained_assignment = None

data_dir = os.path.expanduser("/content/.dataset/bgl/")
output_dir = "../output/bgl/"
log_file = "BGL.log"


def deeplog_file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')


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


def deeplog_file_generator_with_freq(filename, df):
    token_file = filename
    freq_file = filename + "_freq"

    with open(token_file, 'w') as f_tok, open(freq_file, 'w') as f_freq:
        for _, row in df.iterrows():
            seq = row["EventId"]
            f_tok.write(' '.join([str(v) for v in seq]) + '\n')
            freq_seq = rle_count_seq(seq)
            f_freq.write(' '.join([str(v) for v in freq_seq]) + '\n')


def write_sequence_files(filename, sequences):
    deeplog_file_generator_with_freq(filename, pd.DataFrame({"EventId": sequences}))


def parse_log(input_dir, output_dir, log_file, parser_type, ablation):
    log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
    regex = [
        r'(0x)[0-9a-fA-F]+',
        r'\d+.\d+.\d+.\d+',
        r'\d+'
    ]
    if is_semparser_like(ablation):
        regex.insert(0, r'\s+')

    keep_para = False
    if parser_type == "drain":
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=3, st=0.3, rex=regex, keep_para=keep_para)
    elif parser_type == "spell":
        parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=0.55, rex=regex, keep_para=keep_para)
    elif parser_type == "iplom":
        parser = IPLoM.LogParser(log_format=log_format, indir=input_dir, outdir=output_dir, rex=regex)
    else:
        raise ValueError(f"Unsupported parser type: {parser_type}")
    parser.parse(log_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_ablation_argument(parser)
    args = parser.parse_args()

    parse_log(data_dir, output_dir, log_file, 'drain', args.ablation)

    window_size = 5
    step_size = 1
    train_ratio = 0.4

    df = pd.read_csv(f'{output_dir}{log_file}_structured.csv')
    df['datetime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    df['timestamp'] = df["datetime"].values.astype("int64") // 10 ** 9
    df['deltaT'] = df['datetime'].diff() / pd.Timedelta(seconds=1)
    df['deltaT'] = df['deltaT'].fillna(0)
    if is_semantic_id_like(args.ablation):
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
    test_normal = df_normal[train_len:]

    if is_semantic_id_like(args.ablation):
        write_sequence_files(os.path.join(output_dir, "train"), train["EventId"].tolist())
        write_sequence_files(os.path.join(output_dir, "test_normal"), test_normal["EventId"].tolist())
    elif args.ablation == "freq":
        deeplog_file_generator_with_freq(os.path.join(output_dir, 'train'), train)
        deeplog_file_generator_with_freq(os.path.join(output_dir, 'test_normal'), test_normal)
    else:
        deeplog_file_generator(os.path.join(output_dir, 'train'), train, ["EventId"])
        deeplog_file_generator(os.path.join(output_dir, 'test_normal'), test_normal, ["EventId"])

    print("training size {}".format(train_len))
    print("test normal size {}".format(normal_len - train_len))

    del df_normal
    del train
    del test_normal
    gc.collect()

    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    if is_semantic_id_like(args.ablation):
        write_sequence_files(os.path.join(output_dir, "test_abnormal"), df_abnormal["EventId"].tolist())
    elif args.ablation == "freq":
        deeplog_file_generator_with_freq(os.path.join(output_dir, 'test_abnormal'), df_abnormal)
    else:
        deeplog_file_generator(os.path.join(output_dir, 'test_abnormal'), df_abnormal, ["EventId"])
    print('test abnormal size {}'.format(len(df_abnormal)))
