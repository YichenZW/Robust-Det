import os
import json
import nltk
import tiktoken
from transformers import RobertaTokenizer
import random
import torch
import tqdm
from torch.utils.data import TensorDataset
CACHE_DIR = "/cache"

def load_jsonl(path):
    data = []
    with open(path) as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        item = json.loads(line)
        item['dataset_id'] = idx
        data.append(item)
    print("     Successfully loaded {} lines".format(len(data)))
    return data

def load_csv(path):
    import csv
    data = []
    print(f"*Loading data from {path}.")
    with open(path, 'r') as file:
        csv_reader = csv.DictReader(file, delimiter='|')
        for row in csv_reader:
            data.append(row)
    print("*Success. {} Lines Loaded.".format(len(data)))
    return data

def find_closest_sublist(tgt_len, l):
    n = len(l)
    closest_sum = float('inf')
    closest_index = (0, 1)
    for i in range(n):
        sublist_sum = l[i]
        j = i + 1
        while j < n:
            if abs(sublist_sum - tgt_len) < abs(closest_sum - tgt_len):
                closest_sum = sublist_sum
                closest_index = (i, j)
            if sublist_sum == tgt_len:
                return closest_index
            elif sublist_sum < tgt_len:
                sublist_sum += l[j]
                j += 1
            else:
                break
    return closest_index

def trunc_text(text, trg_len):
    sentences = nltk.sent_tokenize(text)
    sen_len = [get_token_numbers(s) for s in sentences]
    sublist = find_closest_sublist(trg_len, sen_len)
    trunced = sentences[sublist[0]: sublist[1]]
    trunced = " ".join(trunced)
    return trunced

os.environ["TIKTOKEN_CACHE_DIR"] = CACHE_DIR
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
def get_token_numbers(t):
    return len(tokenizer.encode(t))
def num_tokens_from_messages(messages):
    """Returns the number of tokens used by a list of messages."""
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(tokenizer.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir=CACHE_DIR)
def get_roberta_token_numbers(t):
    return len(roberta_tokenizer(t)['input_ids'])

def to_tensor_dataset(args, data, tokenizer):
    pad_token = tokenizer.pad_token_id
    if pad_token == None:
        raise ZeroDivisionError

    labels = torch.stack([torch.tensor([int(d['label'])], dtype=torch.float32) for d in data]).squeeze()

    all_input_ids, all_attention_masks = [], []
    for d in data:
        inputs = tokenizer(d['sequence'])
        input_ids, attention_masks = inputs['input_ids'], inputs['attention_mask']
        padding_length = args.max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_masks = attention_masks + ([0] * padding_length)
        input_ids = input_ids[:args.max_seq_length]
        attention_masks = attention_masks[:args.max_seq_length]

        assert len(input_ids) == args.max_seq_length, "Error with input length {} vs {}".format(len(input_ids), args.max_seq_length)
        assert len(attention_masks) == args.max_seq_length, "Error with input length {} vs {}".format(len(attention_masks), args.max_seq_length) 

        all_input_ids.append(input_ids)    
        all_attention_masks.append(attention_masks)
    
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.int)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.int)
    dataset = TensorDataset(labels, all_input_ids, all_attention_masks)
    return dataset

def rand_throw(text, target_len):
    
    sentences = nltk.sent_tokenize(text)
    length = get_token_numbers(text)
    former_len = length
    while length > target_len:
        delete_sent = random.choice(sentences)
        sentences.remove(delete_sent)
        length -= get_token_numbers(delete_sent)
    text = " ".join(sentences)
    print.info("***shorten {} raw tokens into {} tokens.".format(former_len, length))
    return text
def rand_throw_abs(text, target_len):
    # can be slightly longer than target
    sentences = nltk.sent_tokenize(text)
    length = get_token_numbers(text)
    former_len = length
    while length > target_len:
        old_sentences = sentences
        delete_sent = random.choice(sentences)
        sentences.remove(delete_sent)
        new_length = length - get_token_numbers(delete_sent)
        if new_length < target_len:
            if abs(new_length - target_len) < abs(length - target_len):
                length = new_length
                break
            else:
                sentences = old_sentences
                break
        length -= get_token_numbers(delete_sent)
    text = " ".join(sentences)
    print("***shorten {} raw tokens into {} tokens.".format(former_len, length))
    return text

def number_h(num):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, 'Yi')


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def compute_metrics(preds, labels, num_label = 2):
    assert len(preds) == len(labels)
    if num_label == 1: 
        align_preds = []
        given_labels = [0.0, 0.5, 1.0]
        for p in preds:
            temp = np.argmin([abs(g - p) for g in given_labels])
            temp = given_labels[temp]
            align_preds.append(temp)
        # convert into int.
        le = preprocessing.LabelEncoder()
        le.fit(given_labels)
        final_preds = le.transform(align_preds)
        labels = le.transform(labels)

        acc = accuracy_score(final_preds, labels)
        partial = 1 - align_preds.count(0.5) / len(align_preds)
        true_par, false_neu, major_err = 0, 0, 0
        for p, l in zip(final_preds, labels):
            if p != le.transform([0.5]).item() and p == l:
                true_par += 1
            if p == le.transform([0.5]).item() and l != le.transform([0.5]).item():
                false_neu += 1
            if (p == le.transform([0]).item() and l == le.transform([1]).item()) or (p == le.transform([1]).item() and l == le.transform([0]).item()):
                major_err += 1
        return {
            "accuracy": acc,
            "partial": partial,
            "true_partical": true_par / len(align_preds),
            "false_neutral": false_neu / len(align_preds),
            "major_error": major_err / len(align_preds),
        }

    if num_label == 2:
        preds = [round(num) for num in preds]
        acc = accuracy_score(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        fpr, tpr, _ = roc_curve(preds, labels)
        roc_auc = auc(fpr, tpr)
        precision, recall, _  = precision_recall_curve(preds, labels)
        pr_auc = auc(recall, precision)
        return {
            "acc": acc,
            "f1": f1,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_auc": float(roc_auc),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "pr_auc": float(pr_auc),
        }

def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

def round_to_zero_or_one(value, thres, reverse=False):
    if not reverse:
        if value > thres:
            return 1
        elif value == thres:
            return random.randint(0, 1)
        else:
            return 0
    else:
        if value < thres:
            return 1
        elif value == thres:
            return random.randint(0, 1)
        else:
            return 0

def get_overall_metrics(predictions, do_reverse):
    # given the prediction, return metrics. do_reverse=True if human-writen samples are positive labelled.
    # Caution: This setup was changed on October 19th. Now defaultly, human-writen samples "real" are negative label.
    preds_raw = predictions['real'] + predictions['samples']
    thres = np.median(preds_raw)
    preds = [round_to_zero_or_one(p, thres, reverse=do_reverse) for p in preds_raw]


    thres_95 = np.percentile(predictions['real'], 95)
    preds_95 = [round_to_zero_or_one(p, thres_95, reverse=do_reverse) for p in preds_raw]

    labels = [0] * len( predictions['real']) + [1] * len(predictions['samples'])
    acc = accuracy_score(preds, labels)
    asr = sum([ (p==0 and l==1) for p, l in zip(preds, labels)])/len(predictions['samples']) # perceptage in MGTs that predicted as HWTs
    
    acc_95 = accuracy_score(preds_95, labels) 
    acc_95_m = accuracy_score([round_to_zero_or_one(p, thres_95, reverse=do_reverse) for p in predictions['samples']], [1] * len(predictions['samples'])) 
    asr_95 = sum([ (p==0 and l==1) for p, l in zip(preds_95, labels)])/len(predictions['samples']) # perceptage in MGTs that predicted as HWTs

    f1 = f1_score(y_true=labels, y_pred=preds)
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    desired_fpr = 0.05
    index = np.argmin(np.abs([fpr_item - desired_fpr for fpr_item in fpr]))
    tpr_at_fpr5 = tpr[index]
    desired_fpr = 0.10
    index = np.argmin(np.abs([fpr_item - desired_fpr for fpr_item in fpr]))
    tpr_at_fpr10 = tpr[index]
    desired_fpr = 0.20
    index = np.argmin(np.abs([fpr_item - desired_fpr for fpr_item in fpr]))
    tpr_at_fpr20 = tpr[index]

    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    return {"acc": acc, "asr": asr, "acc_95": acc_95, "acc_95_m": acc_95_m, "asr_95": asr_95, "f1": f1, "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc, "p": p, "r": r, "pr_auc": pr_auc, "tpr_at_fpr5": tpr_at_fpr5, "tpr_at_fpr10": tpr_at_fpr10, "tpr_at_fpr20": tpr_at_fpr20}

def histogram_word(data, bins=200,logger=None):
    hist, edges = np.histogram(data, bins=bins, range=(0,1))
    bin_widths = edges[1:] - edges[:-1]
    # logger.info("Histogram: currently ignore zeros.")
    for count, width in zip(hist, bin_widths):
        percent = 100.0 * count / len(data)
        if percent != 0:
            if logger==None:
                print(f"{edges[0]:.4f} - {edges[0]+width:.4f}: {percent:.4f}%")
            else:
                logger.info(f"{edges[0]:.4f} - {edges[0]+width:.4f}: {percent:.4f}%")
        edges = edges[1:]

# *** Utils from MGTBench ***
import transformers
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds\n\n')
        return result
    return timeit_wrapper


# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


def filter_test_data(data, max_length=25):
    new_test = {
        'text': [],
        'label': [],
    }
    for i in range(len(data['test']['text'])):
        text = data['test']['text'][i]
        label = data['test']['label'][i]
        if len(text.split()) <= 25:
            new_test['text'].append(text)
            new_test['label'].append(label)
    data['test'] = new_test
    return data


def load_base_model_and_tokenizer(name, cache_dir):

    print(f'Loading BASE model {name}...')
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        name, cache_dir=cache_dir)
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(
        name, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer


def load_base_model(base_model, DEVICE):
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()

    base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def cal_metrics(label, pred_label, pred_posteriors):
    acc = accuracy_score(label, pred_label)
    precision = precision_score(label, pred_label)
    recall = recall_score(label, pred_label)
    f1 = f1_score(label, pred_label)
    auc = roc_auc_score(label, pred_posteriors)
    return acc, precision, recall, f1, auc


def get_clf_results(x_train, y_train, x_test, y_test):

    clf = LogisticRegression(random_state=0, verbose=1).fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_train_pred_prob = clf.predict_proba(x_train)
    y_train_pred_prob = [_[1] for _ in y_train_pred_prob]
    acc_train, precision_train, recall_train, f1_train, auc_train = cal_metrics(
        y_train, y_train_pred, y_train_pred_prob)
    train_res = acc_train, precision_train, recall_train, f1_train, auc_train

    y_test_pred = clf.predict(x_test)
    y_test_pred_prob = clf.predict_proba(x_test)
    y_test_pred_prob = [_[1] for _ in y_test_pred_prob]
    acc_test, precision_test, recall_test, f1_test, auc_test = cal_metrics(
        y_test, y_test_pred, y_test_pred_prob)
    test_res = acc_test, precision_test, recall_test, f1_test, auc_test

    return train_res, test_res