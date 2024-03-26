from collections import Counter
import torch
import re
device = 'cuda' if torch.cuda.is_available() else "cpu"


def build_bigrams(input_ids):
    bigrams = []
    for i in range(len(input_ids) - 1):
        bigram = tuple(input_ids[i:i+2].tolist())
        bigrams.append(bigram)
    return bigrams

def extract_list(text):
    p = re.compile("^[0-9]+[.)\]\*·:] (.*(?:\n(?![0-9]+[.)\]\*·:]).*)*)", re.MULTILINE)
    return p.findall(text)


def compare_bigram_overlap(input_bigram, para_bigram):
    input_c = Counter(input_bigram)
    para_c = Counter(para_bigram)
    intersection = list(input_c.keys() & para_c.keys())
    overlap = 0
    for i in intersection:
        overlap += input_c[i]
    return overlap


def accept_by_bigram_overlap(sent, para_sents, tokenizer):
    def tokenize(tokenizer, text):
        return tokenizer(text, return_tensors='pt').input_ids[0].to(device)

    input_ids = tokenize(tokenizer, sent)
    input_bigram = build_bigrams(input_ids)

    para_ids = [tokenize(tokenizer, para) for para in para_sents]
    para_bigrams = [build_bigrams(para_id) for para_id in para_ids]
    min_overlap = len(input_ids)
    paraphrased = para_sents[0]
    for i in range(len(para_bigrams)):
        para_bigram = para_bigrams[i]
        overlap = compare_bigram_overlap(input_bigram, para_bigram)
        if overlap < min_overlap and len(para_ids[i]) <= 1.5 * len(input_ids):
            min_overlap = overlap
            paraphrased = para_sents[i]
    return paraphrased


