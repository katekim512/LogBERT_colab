import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import torch
from tqdm import tqdm
tqdm.disable = True
from torch.utils.data import DataLoader
import gc

from bert_pytorch.dataset import WordVocab
from bert_pytorch.dataset import LogDataset
from bert_pytorch.dataset.sample import fixed_window


def compute_anomaly(results, params, seq_threshold=0.5):
    is_logkey = params["is_logkey"]
    is_time = params["is_time"]
    total_errors = 0
    for seq_res in results:
        # label pairs as anomaly when over half of masked tokens are undetected
        if (is_logkey and seq_res["undetected_tokens"] > seq_res["masked_tokens"] * seq_threshold) or \
                (is_time and seq_res["num_error"]> seq_res["masked_tokens"] * seq_threshold) or \
                (params["hypersphere_loss_test"] and seq_res["deepSVDD_label"]):
            total_errors += 1
    return total_errors


def find_best_threshold(test_normal_results, test_abnormal_results, params, th_range, seq_range):
    best_result = [0] * 9
    for seq_th in seq_range:
        FP = compute_anomaly(test_normal_results, params, seq_th)
        TP = compute_anomaly(test_abnormal_results, params, seq_th)

        if TP == 0:
            continue

        TN = len(test_normal_results) - FP
        FN = len(test_abnormal_results) - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)

        if F1 > best_result[-1]:
            best_result = [0, seq_th, FP, TP, TN, FN, P, R, F1]
    return best_result


class Predictor():
    def __init__(self, options):
        self.model_path = options["model_path"]
        self.vocab_path = options["vocab_path"]
        self.device = options["device"]
        self.window_size = options["window_size"]
        self.adaptive_window = options["adaptive_window"]
        self.seq_len = options["seq_len"]
        self.corpus_lines = options["corpus_lines"]
        self.on_memory = options["on_memory"]
        self.batch_size = options["batch_size"]
        self.num_workers = options["num_workers"]
        self.num_candidates = options["num_candidates"]
        self.output_dir = options["output_dir"]
        self.model_dir = options["model_dir"]
        self.gaussian_mean = options["gaussian_mean"]
        self.gaussian_std = options["gaussian_std"]

        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.scale_path = options["scale_path"]

        self.hypersphere_loss = options["hypersphere_loss"]
        self.hypersphere_loss_test = options["hypersphere_loss_test"]

        self.lower_bound = self.gaussian_mean - 3 * self.gaussian_std
        self.upper_bound = self.gaussian_mean + 3 * self.gaussian_std

        self.center = None
        self.radius = None
        self.test_ratio = options["test_ratio"]
        self.mask_ratio = options["mask_ratio"]
        self.min_len=options["min_len"]

    # def detect_logkey_anomaly(self, masked_output, masked_label):
    #     num_undetected_tokens = 0
    #     output_maskes = []
    #     for i, token in enumerate(masked_label):
    #         # output_maskes.append(torch.argsort(-masked_output[i])[:30].cpu().numpy()) # extract top 30 candidates for mask labels

    #         if token not in torch.argsort(-masked_output[i])[:self.num_candidates]:
    #             num_undetected_tokens += 1

    #     return num_undetected_tokens, [output_maskes, masked_label.cpu().numpy()]

    def detect_logkey_anomaly(self, masked_output, masked_label):
        # 기존: 토큰마다 argsort 개별 호출 → 느리고 CPU 전송 반복
        top_k = torch.argsort(-masked_output, dim=1)[:, :self.num_candidates]  # 한 번에 처리
        match = (top_k == masked_label.unsqueeze(1)).any(dim=1)
        num_undetected_tokens = (~match).sum().item()
        return num_undetected_tokens, [[], masked_label.cpu().numpy()]


    @staticmethod
    def generate_test(output_dir, file_name, window_size, adaptive_window, seq_len, scale, min_len, chunk_size=5000):
        """
        :return: log_seqs: num_samples x session(seq)_length, tim_seqs: num_samples x session_length
        """
        log_seqs = []
        tim_seqs = []
        with open(output_dir + file_name, "r") as f:
            for idx, line in tqdm(enumerate(f)):
                #if idx > 40: break
                log_seq, tim_seq = fixed_window(line, window_size,
                                                adaptive_window=adaptive_window,
                                                seq_len=seq_len, min_len=min_len)
                if len(log_seq) == 0:
                    continue

                # if scale is not None:
                #     times = tim_seq
                #     for i, tn in enumerate(times):
                #         tn = np.array(tn).reshape(-1, 1)
                #         times[i] = scale.transform(tn).reshape(-1).tolist()
                #     tim_seq = times

                log_seqs += log_seq
                tim_seqs += tim_seq

                if len(log_seqs) >= chunk_size:
                    yield log_seqs[:chunk_size], tim_seqs[:chunk_size]
                    log_seqs = log_seqs[chunk_size:]
                    tim_seqs = tim_seqs[chunk_size:]

        if log_seqs:
            yield log_seqs, tim_seqs

        # sort seq_pairs by seq len
        fixed = []
        fixed_log = []
        fixed_time = []

        for log_seq, time_seq in zip(log_seqs, tim_seqs):

            log_seq = list(log_seq)
            time_seq = list(time_seq)

            if len(log_seq) >= window_size:
                log_seq = log_seq[:window_size]
                time_seq = time_seq[:window_size]

            else:
                pad_len = window_size - len(log_seq)

                log_seq = log_seq + [0] * pad_len
                time_seq = time_seq + [0] * pad_len

            fixed_log.append(log_seq)
            fixed_time.append(time_seq)

        log_seqs = np.array(fixed_log)
        tim_seqs = np.array(fixed_time)

        test_len = list(map(len, log_seqs))
        test_sort_index = np.argsort(-1 * np.array(test_len))

        log_seqs = log_seqs[test_sort_index]
        tim_seqs = tim_seqs[test_sort_index]

        print(f"{file_name} size: {len(log_seqs)}")
        return log_seqs, tim_seqs

    # def helper(self, model, output_dir, file_name, vocab, scale=None, error_dict=None):
    #     total_results = []
    #     total_errors = []
    #     output_results = []
    #     total_dist = []
    #     output_cls = []

    #     part = 0

    #     logkey_test, time_test = self.generate_test(output_dir, file_name, self.window_size, self.adaptive_window, self.seq_len, scale, self.min_len)

    #     # use 1/10 test data
    #     if self.test_ratio != 1:
    #         num_test = len(logkey_test)
    #         rand_index = torch.randperm(num_test)
    #         rand_index = rand_index[:int(num_test * self.test_ratio)] if isinstance(self.test_ratio, float) else rand_index[:self.test_ratio]
    #         logkey_test, time_test = logkey_test[rand_index], time_test[rand_index]


    #     seq_dataset = LogDataset(logkey_test, time_test, vocab, seq_len=self.seq_len,
    #                              corpus_lines=self.corpus_lines, on_memory=self.on_memory, predict_mode=True, mask_ratio=self.mask_ratio)

    #     # use large batch size in test data
    #     data_loader = DataLoader(seq_dataset, batch_size=self.batch_size, num_workers=1,
    #                              collate_fn=seq_dataset.collate_fn)

    #     for idx, data in enumerate(data_loader):
    #         data = {key: value.to(self.device) for key, value in data.items()}

    #         result = model(data["bert_input"], data["time_input"])

    #         # mask_lm_output, mask_tm_output: batch_size x session_size x vocab_size
    #         # cls_output: batch_size x hidden_size
    #         # bert_label, time_label: batch_size x session_size
    #         # in session, some logkeys are masked

    #         mask_lm_output, mask_tm_output = result["logkey_output"], result["time_output"]
    #         # output_cls += result["cls_output"].tolist()

    #         # dist = torch.sum((result["cls_output"] - self.hyper_center) ** 2, dim=1)
    #         # when visualization no mask
    #         # continue

    #         # loop though each session in batch
    #         for i in range(len(data["bert_label"])):
    #             seq_results = {"num_error": 0,
    #                            "undetected_tokens": 0,
    #                            "masked_tokens": 0,
    #                            "total_logkey": torch.sum(data["bert_input"][i] > 0).item(),
    #                            "deepSVDD_label": 0
    #                            }

    #             mask_index = data["bert_label"][i] > 0
    #             num_masked = torch.sum(mask_index).tolist()
    #             seq_results["masked_tokens"] = num_masked

    #             if self.is_logkey:
    #                 num_undetected, output_seq = self.detect_logkey_anomaly(
    #                     mask_lm_output[i][mask_index], data["bert_label"][i][mask_index])
    #                 seq_results["undetected_tokens"] = num_undetected

    #                 output_results.append(output_seq)

    #             if self.hypersphere_loss_test:
    #                 # detect by deepSVDD distance
    #                 assert result["cls_output"][i].size() == self.center.size()
    #                 # dist = torch.sum((result["cls_fnn_output"][i] - self.center) ** 2)
    #                 dist = torch.sqrt(torch.sum((result["cls_output"][i] - self.center) ** 2))
    #                 #total_dist.append(dist.item())

    #                 # user defined threshold for deepSVDD_label
    #                 seq_results["deepSVDD_label"] = int(dist.item() > self.radius)
    #                 #
    #                 # if dist > 0.25:
    #                 #     pass

    #             if idx < 10 or idx % 1000 == 0:
    #                 print(
    #                     "{}, #time anomaly: {} # of undetected_tokens: {}, # of masked_tokens: {} , "
    #                     "# of total logkey {}, deepSVDD_label: {} \n".format(
    #                         file_name,
    #                         seq_results["num_error"],
    #                         seq_results["undetected_tokens"],
    #                         seq_results["masked_tokens"],
    #                         seq_results["total_logkey"],
    #                         seq_results['deepSVDD_label']
    #                     )
    #                 )
    #             total_results.append(seq_results)

    #             if len(total_results) >= 10000:
    #                 save_path = self.model_dir + f"{file_name}_results_part_{part:03d}.pkl"

    #                 with open(save_path, "wb") as f:
    #                     pickle.dump(total_results, f)

    #                 print(f"Saved chunk {part} → {save_path}")

    #                 total_results = []
    #                 part += 1
    #                 output_results = []
    #                 total_dist = []
    #                 gc.collect()

    #     if len(total_results) > 0:
    #         save_path = self.model_dir + f"{file_name}_results_part_{part:03d}.pkl"

    #         with open(save_path, "wb") as f:
    #             pickle.dump(total_results, f)

    #         print(f"Saved final chunk {part} → {save_path}")
    #     # for time
    #     # return total_results, total_errors

    #     #for logkey
    #     # return total_results, output_result

    #     # for hypersphere distance
    #     return total_results, []

    def helper(self, model, output_dir, file_name, vocab, scale=None, error_dict=None):
        total_results = []
        part = 0

        for log_chunk, tim_chunk in self.generate_test(output_dir, file_name,
                                                    self.window_size, self.adaptive_window,
                                                    self.seq_len, scale, self.min_len,
                                                    chunk_size=5000):
            fixed_log, fixed_time = [], []
            for log_seq, time_seq in zip(log_chunk, tim_chunk):
                log_seq, time_seq = list(log_seq), list(time_seq)
                if len(log_seq) >= self.window_size:
                    log_seq = log_seq[:self.window_size]
                    time_seq = time_seq[:self.window_size]
                else:
                    pad_len = self.window_size - len(log_seq)
                    log_seq = log_seq + [0] * pad_len
                    time_seq = time_seq + [0] * pad_len
                fixed_log.append(log_seq)
                fixed_time.append(time_seq)

            logkey_test = np.array(fixed_log)
            tim_test = np.array(fixed_time)

            seq_dataset = LogDataset(logkey_test, tim_test, vocab, seq_len=self.seq_len,
                                 corpus_lines=self.corpus_lines, on_memory=self.on_memory,
                                 predict_mode=True, mask_ratio=self.mask_ratio)

            data_loader = DataLoader(seq_dataset, batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 collate_fn=seq_dataset.collate_fn)

            for idx, data in enumerate(data_loader):
                data = {key: value.to(self.device) for key, value in data.items()}

                with torch.no_grad():
                    result = model(data["bert_input"], data["time_input"])

                mask_lm_output = result["logkey_output"]

                for i in range(len(data["bert_label"])):
                    seq_results = {"num_error": 0, "undetected_tokens": 0,
                               "masked_tokens": 0,
                               "total_logkey": torch.sum(data["bert_input"][i] > 0).item(),
                               "deepSVDD_label": 0}

                    mask_index = data["bert_label"][i] > 0
                    seq_results["masked_tokens"] = torch.sum(mask_index).tolist()

                    if self.is_logkey:
                        num_undetected, _ = self.detect_logkey_anomaly(
                            mask_lm_output[i][mask_index], data["bert_label"][i][mask_index])
                        seq_results["undetected_tokens"] = num_undetected

                    if self.hypersphere_loss_test:
                        dist = torch.sqrt(torch.sum((result["cls_output"][i] - self.center) ** 2))
                        seq_results["deepSVDD_label"] = int(dist.item() > self.radius)

                    total_results.append(seq_results)

                    if len(total_results) >= 10000:
                        save_path = self.model_dir + f"{file_name}_results_part_{part:03d}.pkl"
                        with open(save_path, "wb") as f:
                            pickle.dump(total_results, f)
                        print(f"Saved chunk {part} → {save_path}")
                        total_results = []
                        part += 1
                        gc.collect()

            del logkey_test, tim_test, seq_dataset, data_loader
            gc.collect()

        if total_results:
            save_path = self.model_dir + f"{file_name}_results_part_{part:03d}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(total_results, f)
            print(f"Saved final chunk {part} → {save_path}")

        all_results = []
        for i in range(part + 1):
            path = self.model_dir + f"{file_name}_results_part_{i:03d}.pkl"
            with open(path, "rb") as f:
                all_results += pickle.load(f)

        return all_results, []

    def predict(self):
        model = torch.load(self.model_path, weights_only=False, map_location=self.device)
        model.to(self.device)
        model.eval()
        print('model_path: {}'.format(self.model_path))

        start_time = time.time()
        vocab = WordVocab.load_vocab(self.vocab_path)

        scale = None
        error_dict = None
        if self.is_time:
            with open(self.scale_path, "rb") as f:
                scale = pickle.load(f)

            with open(self.model_dir + "error_dict.pkl", 'rb') as f:
                error_dict = pickle.load(f)

        if self.hypersphere_loss:
            center_dict = torch.load(self.model_dir + "best_center.pt", weights_only=False)
            self.center = center_dict["center"]
            self.radius = center_dict["radius"]
            # self.center = self.center.view(1,-1)

        print("test normal predicting")
        test_normal_results, test_normal_errors = self.helper(model, self.output_dir, "test_normal", vocab, scale, error_dict)
        print("test normal END!!")

        print("test abnormal predicting")
        test_abnormal_results, test_abnormal_errors = self.helper(model, self.output_dir, "test_abnormal", vocab, scale, error_dict)

        print("Saving test normal results")
        with open(self.model_dir + "test_normal_results", "wb") as f:
            pickle.dump(test_normal_results, f)

        print("Saving test abnormal results")
        with open(self.model_dir + "test_abnormal_results", "wb") as f:
            pickle.dump(test_abnormal_results, f)

        # print("Saving test normal errors")
        # with open(self.model_dir + "test_normal_errors.pkl", "wb") as f:
        #     pickle.dump(test_normal_errors, f)

        # print("Saving test abnormal results")
        # with open(self.model_dir + "test_abnormal_errors.pkl", "wb") as f:
        #     pickle.dump(test_abnormal_errors, f)

        if test_normal_errors is not None:
            print("Saving test normal errors")
            with open(self.model_dir + "test_normal_errors.pkl", "wb") as f:
                pickle.dump(test_normal_errors, f)

        if test_abnormal_errors is not None:
            print("Saving test abnormal errors")
            with open(self.model_dir + "test_abnormal_errors.pkl", "wb") as f:
                pickle.dump(test_abnormal_errors, f)

        params = {"is_logkey": self.is_logkey, "is_time": self.is_time, "hypersphere_loss": self.hypersphere_loss,
                  "hypersphere_loss_test": self.hypersphere_loss_test}
        best_th, best_seq_th, FP, TP, TN, FN, P, R, F1 = find_best_threshold(test_normal_results,
                                                                            test_abnormal_results,
                                                                            params=params,
                                                                            th_range=np.arange(10),
                                                                            seq_range=np.arange(0,1,0.1))

        print("best threshold: {}, best threshold ratio: {}".format(best_th, best_seq_th))
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        print('Precision: {:.2f}%, Recall: {:.2f}%, F1-measure: {:.2f}%'.format(P, R, F1))
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))


