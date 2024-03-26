import sys
import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
import custom_datasets
from multiprocessing.pool import ThreadPool
import time
from utils import load_csv


# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

DEVICE = "cuda"

def tokenize_and_mask(args, text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(args, texts, mask_model, mask_tokenizer):
    if mask_model != "gpt-3.5-turbo":
        n_expected = count_masks(texts)
        stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
        outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
        return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
    else:
        from paraphrase_openai import query_openai
        res = []
        for t in texts:
            t_and_p = t + " \n\n Output the list of insered words: <extra_id_0> inserted word <extra_id_1> inserted word <extra_id_2> inserted word ..."
            res.append(query_openai(t_and_p))
        return res


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(args, texts, mask_model, mask_tokenizer, span_length, pct, ceil_pct=False):
    if mask_model == "gpt-3.5-turbo":
        if not args.random_fills:
            masked_texts = [tokenize_and_mask(args, x, span_length, pct, ceil_pct) for x in texts]
            raw_fills = replace_masks(args, masked_texts, mask_model, mask_tokenizer)
            extracted_fills = extract_fills(raw_fills)
            perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
    else:
        if not args.random_fills:
            masked_texts = [tokenize_and_mask(args, x, span_length, pct, ceil_pct) for x in texts]
            
            raw_fills = replace_masks(args, masked_texts, mask_model, mask_tokenizer)
            
            extracted_fills = extract_fills(raw_fills)

            perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

            attempts = 1
            while '' in perturbed_texts:
                idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
                print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
                masked_texts = [tokenize_and_mask(args, x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
                raw_fills = replace_masks(args, masked_texts, mask_model, mask_tokenizer)
                extracted_fills = extract_fills(raw_fills)
                new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
                for idx, x in zip(idxs, new_perturbed_texts):
                    perturbed_texts[idx] = x
                attempts += 1
        else:
            raise NotImplementedError
        
    return perturbed_texts


def perturb_texts(args, texts, masked_model, masked_tokenizer, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size
    if '11b' in args.mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(args, texts[i:i + chunk_size], masked_model, masked_tokenizer, span_length, pct, ceil_pct=ceil_pct))
    return outputs


def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])
