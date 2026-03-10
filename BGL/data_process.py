import sys
sys.path.append('../')

import os
import gc
import pandas as pd
import numpy as np
from logparser import Spell, Drain, IPLoM
import argparse
from tqdm import tqdm
from logdeep.dataset.session import sliding_window

tqdm.pandas()
pd.options.mode.chained_assignment = None

PAD = 0
UNK = 1
START = 2

data_dir = os.path.expanduser("/content/.dataset/bgl/") 
output_dir = "../output/bgl/" 
log_file = "BGL.log"

def compress_log_storm(df, time_threshold=0.5, keep_ratio=0.5):
    """
    연속된 중복 로그를 무조건 1개로 줄이는 대신, 일정 비율로 압축합니다.
    - keep_ratio: 0.5이면 중복 묶음 중 50%를 무작위로 남깁니다.
    """
    if df.empty:
        return df

    # 1. 중복 그룹 식별 (EventId가 바뀌거나 시간차가 크면 새로운 그룹)
    df['time_diff'] = df['datetime'].diff().dt.total_seconds()
    df['is_new_group'] = (df['EventId'] != df['EventId'].shift(1)) | (df['time_diff'] > time_threshold)
    df['group_id'] = df['is_new_group'].cumsum()

    # 2. 그룹별 샘플링 수행
    def sample_group(group):
        if len(group) <= 1: # 중복이 없는 경우 그대로 유지
            return group
        # 그룹 내에서 지정된 비율만큼 샘플링 (최소 1개는 유지)
        n_samples = max(1, int(len(group) * keep_ratio))
        return group.sample(n=n_samples).sort_index()

    # 그룹화하여 샘플링 적용
    df_compressed = df.groupby('group_id', group_keys=False).apply(sample_group)

    # 3. 임시 컬럼 제거 및 뒷정리
    df_compressed = df_compressed.drop(columns=['time_diff', 'is_new_group', 'group_id'])
    return df_compressed.reset_index(drop=True)

# In the first column of the log, "-" indicates non-alert messages while others are alert messages.
def count_anomaly():
    total_size = 0
    normal_size = 0
    with open(data_dir + log_file, encoding="utf8") as f:
        for line in f:
            total_size += 1
            if line.split(' ',1)[0] == '-':
                normal_size += 1
    print("total size {}, abnormal size {}".format(total_size, total_size - normal_size))


# def deeplog_df_transfer(df, features, target, time_index, window_size):
#     """
#     :param window_size: offset datetime https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
#     :return:
#     """
#     agg_dict = {target:'max'}
#     for f in features:
#         agg_dict[f] = _custom_resampler
#
#     features.append(target)
#     features.append(time_index)
#     df = df[features]
#     deeplog_df = df.set_index(time_index).resample(window_size).agg(agg_dict).reset_index()
#     return deeplog_df
#
#
# def _custom_resampler(array_like):
#     return list(array_like)


def deeplog_file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')


def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
    regex = [
        r'\s+',
        r'(0x)[0-9a-fA-F]+', #hexadecimal
        r'\d+.\d+.\d+.\d+',
        # r'/\w+( )$'
        r'\d+'
    ]
    keep_para = False
    if parser_type == "drain":
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.3  # Similarity threshold
        depth = 3  # Depth of all leaf nodes
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=keep_para)
        parser.parse(log_file)
    elif parser_type == "spell":
        tau = 0.55
        parser = Spell.LogParser(indir=data_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=keep_para)
        parser.parse(log_file)
    elif parser_type == "iplom":
        parser = IPLoM.LogParser(log_format=log_format, indir=input_dir, outdir=output_dir, rex=regex)
        parser.parse(log_file)

#
# def merge_list(time, activity):
#     time_activity = []
#     for i in range(len(activity)):
#         temp = []
#         assert len(time[i]) == len(activity[i])
#         for j in range(len(activity[i])):
#             temp.append(tuple([time[i][j], activity[i][j]]))
#         time_activity.append(np.array(temp))
#     return time_activity


if __name__ == "__main__":
    #
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p', default=None, type=str, help="parser type")
    # parser.add_argument('-w', default='T', type=str, help='window size(mins)')
    # parser.add_argument('-s', default='1', type=str, help='step size(mins)')
    # parser.add_argument('-r', default=0.4, type=float, help="train ratio")
    # args = parser.parse_args()
    # print(args)
    #

    ##########
    # Parser #
    #########

    #parse_log(data_dir, output_dir, log_file, 'drain')

    #########
    # Count #
    #########
    # count_anomaly()

    ##################
    # Transformation #
    ##################
    window_size = 5
    step_size = 1
    train_ratio = 0.4
    time_threshold = 0.5 

    # [실험 설정] 아래 모드 중 하나를 선택하세요:
    # 'ratio_0.0' (완전제거), 'ratio_0.3', 'ratio_0.5', 'ratio_0.7'
    experiment_mode = 'ratio_0.5' 

    if experiment_mode == 'ratio_0.0':
        keep_ratio = 0.0  # 사실상 1개만 남김
        print("Mode: Full Deduplication (Keep only 1)")
    else:
        keep_ratio = float(experiment_mode.split('_')[1])
        print(f"Mode: Proportional Compression (Ratio: {keep_ratio})")

    df = pd.read_csv(f'{output_dir}{log_file}_structured.csv')

    # data preprocess
    df['datetime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))

    #################################################################
    # [수정] Train/Test 분할 후 모드별 압축 로직
    #################################################################
    
    split_index = int(len(df) * train_ratio)
    df_train_raw = df.iloc[:split_index].copy()
    df_test_raw = df.iloc[split_index:].copy()

    print(f"Train (Before): {len(df_train_raw)} lines")
    
    # 중복 그룹 식별
    df_train_raw['time_diff'] = df_train_raw['datetime'].diff().dt.total_seconds()
    df_train_raw['is_new'] = (df_train_raw['EventId'] != df_train_raw['EventId'].shift(1)) | \
                             (df_train_raw['time_diff'] > time_threshold)
    df_train_raw['group_id'] = df_train_raw['is_new'].cumsum()

    # --- 실험 모드별 샘플링 함수 ---
    def apply_sampling(group):
        if len(group) <= 1:
            return group
        
        if experiment_mode == 'ratio_0.0':
            # 무조건 첫 번째 로그만 남김
            return group.head(1)
        else:
            # 설정된 비율만큼 샘플링 (최소 1개 보장)
            n_samples = max(1, int(len(group) * keep_ratio))
            return group.sample(n=n_samples).sort_index()

    # 모드 적용
    df_train = df_train_raw.groupby('group_id', group_keys=False).apply(apply_sampling).reset_index(drop=True)
    
    # 특성 재계산
    df_train = df_train.drop(columns=['time_diff', 'is_new', 'group_id'])
    df_train['timestamp'] = df_train["datetime"].values.astype(np.int64) // 10 ** 9
    df_train['deltaT'] = df_train['datetime'].diff().dt.total_seconds().fillna(0)
    print(f"Train (After {experiment_mode}): {len(df_train)} lines")

    # ---  2. Test 세트: 원본 유지 및 특성 계산 ---
    df_test = df_test_raw.copy().reset_index(drop=True)
    df_test['timestamp'] = df_test["datetime"].values.astype(np.int64) // 10 ** 9
    df_test['deltaT'] = df_test['datetime'].diff().dt.total_seconds().fillna(0)
    print(f"Test (Original): {len(df_test)} lines")

    # ---  3. 슬라이딩 윈도우 생성 ---
    print("Generating sliding windows... (This may take a while)")
    train_deeplog_df = sliding_window(df_train[["timestamp", "Label", "EventId", "deltaT"]],
                                      para={"window_size": int(window_size)*60, "step_size": int(step_size) * 60})
                                      
    test_deeplog_df = sliding_window(df_test[["timestamp", "Label", "EventId", "deltaT"]],
                                     para={"window_size": int(window_size)*60, "step_size": int(step_size) * 60})

    # --- 4. 실험 모드별 경로 정의 및 생성 ---
    final_output_dir = os.path.join(output_dir, experiment_mode)
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
        print(f"Created directory: {final_output_dir}")
    
    # 윈도우 생성 후 원본 df 삭제
    del df_train, df_test, df_train_raw, df_test_raw
    gc.collect()

    #########
    # Train #
    #########
    df_normal_train = train_deeplog_df[train_deeplog_df["Label"] == 0]
    df_normal_train = df_normal_train.sample(frac=1, random_state=12).reset_index(drop=True)
    
    # [수정] output_dir 대신 final_output_dir 사용
    deeplog_file_generator(os.path.join(final_output_dir, 'train'), df_normal_train, ["EventId"])
    print(f"[{experiment_mode}] Final training size: {len(df_normal_train)}")


    ###############
    # Test Normal #
    ###############
    df_normal_test = test_deeplog_df[test_deeplog_df["Label"] == 0]
    # [수정] output_dir 대신 final_output_dir 사용
    deeplog_file_generator(os.path.join(final_output_dir, 'test_normal'), df_normal_test, ["EventId"])
    print(f"[{experiment_mode}] Final test normal size: {len(df_normal_test)}")

    # 메모리 정리
    del df_normal_train
    del df_normal_test
    gc.collect()

    #################
    # Test Abnormal #
    #################
    df_abnormal_test = test_deeplog_df[test_deeplog_df["Label"] == 1]
    # [수정] output_dir 대신 final_output_dir 사용
    deeplog_file_generator(os.path.join(final_output_dir, 'test_abnormal'), df_abnormal_test, ["EventId"])
    print(f"[{experiment_mode}] Final test abnormal size: {len(df_abnormal_test)}")
    
    del df_abnormal_test
    gc.collect()