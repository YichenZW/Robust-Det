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
random.seed(0)
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
import custom_datasets
from multiprocessing.pool import ThreadPool
import time
import openpyxl
from utils import load_csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

TOP_PERCENT = 0.07
print(f"TOP_PERCENT = {TOP_PERCENT}")

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

def load_base_model():
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    try:
        mask_model.cpu()
    except NameError:
        pass
    if args.openai_model is None:
        base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def load_mask_model():
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    if args.openai_model is None:
        base_model.cpu()
    if not args.random_fills:
        mask_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def tokenize_and_mask(text, span_length, pct, ceil_pct=False, pred_tokens=[]):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    if len(pred_tokens) >= 8:
        # print("Too much/diverous prevention.")
        pred_tokens = []
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        retry = False
        for pt in pred_tokens:
            for tks in tokens[search_start:search_end]:
                if pt in tks: # pt is token level but tks is word level
                    retry = True
                    break
        if not retry and mask_string not in tokens[search_start:search_end]:
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
def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


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

def get_next_logits(input_ids, logits, top_percentage=TOP_PERCENT):
    log_likelihoods = [0,]
    for i, logit in enumerate(logits[:-1,:]):
        next_token_id = input_ids[i+1]
        next_logit = logit[next_token_id] - torch.logsumexp(logit, dim=-1)
        log_likelihoods.append(next_logit)

    log_likelihood_tuples = [(log_likelihood, index) for index, log_likelihood in enumerate(log_likelihoods)]

    sorted_tuples = sorted(log_likelihood_tuples, key=lambda x: x[0])

    num_top_elements = int(len(sorted_tuples) * top_percentage)

    top_indices = [index for _, index in sorted_tuples[:num_top_elements]]
    return top_indices

def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    if not args.random_fills:
        pred_tokenses = []
        base_model_ = base_model.to(DEVICE)
        for text in texts:
            with torch.no_grad():
                tokens = base_tokenizer(text, return_tensors="pt").to(DEVICE)
                labels = tokens.input_ids
                logits = base_model_(**tokens, labels=labels).logits
                pred_pos = get_next_logits(labels[0], logits[0])
                pred_tokens = set([base_tokenizer.decode(labels[0][pos_i].item(), skip_special_tokens=True) for pos_i in pred_pos])
                pred_tokenses.append(pred_tokens)

        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct, ptok) for x, ptok in zip(texts, pred_tokenses)]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
    else:
        if args.random_fills_tokens:
            # tokenize base_tokenizer
            tokens = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
            valid_tokens = tokens.input_ids != base_tokenizer.pad_token_id
            replace_pct = args.pct_words_masked * (args.span_length / (args.span_length + 2 * args.buffer_size))

            # replace replace_pct of input_ids with random tokens
            random_mask = torch.rand(tokens.input_ids.shape, device=DEVICE) < replace_pct
            random_mask &= valid_tokens
            random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            # while any of the random tokens are special tokens, replace them with random non-special tokens
            while any(base_tokenizer.decode(x) in base_tokenizer.all_special_tokens for x in random_tokens):
                random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            tokens.input_ids[random_mask] = random_tokens
            perturbed_texts = base_tokenizer.batch_decode(tokens.input_ids, skip_special_tokens=True)
        else:
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
            perturbed_texts = masked_texts
            # replace each <extra_id_*> with args.span_length random words from FILL_DICTIONARY
            for idx, text in enumerate(perturbed_texts):
                filled_text = text
                for fill_idx in range(count_masks([text])[0]):
                    fill = random.sample(FILL_DICTIONARY, span_length)
                    filled_text = filled_text.replace(f"<extra_id_{fill_idx}>", " ".join(fill))
                assert count_masks([filled_text])[0] == 0, "Failed to replace all masks"
                perturbed_texts[idx] = filled_text

    return perturbed_texts


def perturb_texts(texts, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size
    if '11b' in mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs


def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])


def _openai_sample(p):
    if args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = { "engine": args.openai_model, "max_tokens": 200 }
    if args.do_top_p:
        kwargs['top_p'] = args.top_p
    
    r = openai.Completion.create(prompt=f"{p}", **kwargs)
    return p + r['choices'][0].text


# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(texts, min_words=55, prompt_tokens=30):
    # encode each text as a list of token ids
    if args.dataset == 'pubmed':
        texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts]
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    else:
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
        all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

    if args.openai_model:
        # decode the prefixes back into text
        prefixes = base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
        pool = ThreadPool(args.batch_size)

        decoded = pool.map(_openai_sample, prefixes)
    else:
        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < min_words:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs = {}
            if args.do_top_p:
                sampling_kwargs['top_p'] = args.top_p
            elif args.do_top_k:
                sampling_kwargs['top_k'] = args.top_k
            min_length = 50 if args.dataset in ['pubmed'] else 150
            outputs = base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)
            decoded = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1

    if args.openai_model:
        global API_TOKEN_COUNTER

        # count total number of tokens with GPT2_TOKENIZER
        total_tokens = sum(len(GPT2_TOKENIZER.encode(x)) for x in decoded)
        API_TOKEN_COUNTER += total_tokens

    return decoded


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])[:-1]
    labels = labels.view(-1)[1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean()


# Get the log likelihood of each text under the base_model
def get_ll(text): 
    if args.openai_model:        
        kwargs = { "engine": args.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        with torch.no_grad():
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids

            logits = base_model(**tokenized, labels=labels).logits

            pred_pos = get_next_logits(labels[0], logits[0])

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            import torch.nn as nn
            loss_fct = nn.CrossEntropyLoss()
            base_loss = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

            def direct_remove():    
                filtered_labels = torch.stack([element for index, element in enumerate(labels.view(-1)) if index not in pred_pos]).unsqueeze(0)
                viewed_logits = logits.view(-1, logits.size(-1))
                filtered_logits = viewed_logits[torch.tensor([i for i in range(len(viewed_logits)) if i not in pred_pos])]
                shift_filter_logits = filtered_logits[..., :-1, :].contiguous()
                shift_filter_labels = filtered_labels[..., 1:].contiguous()
                filtered_loss = -loss_fct(shift_filter_logits.view(-1, shift_filter_logits.size(-1)), shift_filter_labels.view(-1)).item()
                if filtered_loss > base_loss: 
                # we suggest in such case, this text is attacked
                    return filtered_loss
                else:
                    return base_loss
            def pad_token():
                pad_id = base_tokenizer.eos_token_id
                for t_idx in pred_pos:
                    tokenized.input_ids[0][t_idx] = pad_id
                labels = tokenized.input_ids
                filtered_loss = -base_model(**tokenized, labels=labels).loss.item()
                if filtered_loss > base_loss: 
                # we suggest in such case, this text is attacked
                    return filtered_loss
                else:
                    return base_loss
            def remove_token():
                labels = tokenized.input_ids[0]
                removed_labels = torch.stack([l for t_idx, l in enumerate(labels) if t_idx not in pred_pos])
                removed_text = base_tokenizer.decode(removed_labels, skip_special_tokens=True)
                removed_tokenized = base_tokenizer(removed_text, return_tensors="pt").to(DEVICE)
                filtered_loss = -base_model(**removed_tokenized, labels=removed_tokenized.input_ids).loss.item()
               
                return filtered_loss
            
            return remove_token()

def get_lls(texts):
    if not args.openai_model:
        return [get_ll(text) for text in texts]
    else:
        global API_TOKEN_COUNTER

        # use GPT2_TOKENIZER to get total number of tokens
        total_tokens = sum(len(GPT2_TOKENIZER.encode(text)) for text in texts)
        API_TOKEN_COUNTER += total_tokens * 2  # multiply by two because OpenAI double-counts echo_prompt tokens

        pool = ThreadPool(args.batch_size)
        return pool.map(get_ll, texts)


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, log=False):
    assert args.openai_model is None, "get_rank not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


# get average entropy of each token in the text
def get_entropy(text):
    assert args.openai_model is None, "get_entropy not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

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

# save the ROC curve for each experiment, given a list of output dictionaries, one for each experiment, using colorblind-friendly colors
def save_roc_curves(experiments):
    
    plt.clf()

    for experiment, color in zip(experiments, COLORS):
        metrics = experiment["metrics"]
        pr_metrics = experiment['pr_metrics']
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{experiment['name']}, roc_auc={metrics['roc_auc']:.3f}", color=color)
        # print roc_auc for this experiment
        print(f"{experiment['name']} roc_auc: {metrics['roc_auc']:.5f} pr_auc: {pr_metrics['pr_auc']:.5f} acc: {metrics['acc']:.5f} f1: {metrics['f1']:.5f}")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves ({base_model_name} - {args.mask_filling_model_name})')
    plt.legend(loc="lower right", fontsize=6)
    plt.savefig(f"{SAVE_FOLDER}/roc_curves.png")


# save the histogram of log likelihoods in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_ll_histograms(experiments):
    
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)
            plt.hist([r["sampled_ll"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["perturbed_sampled_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed sampled')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.subplot(1, 2, 2)
            plt.hist([r["original_ll"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.hist([r["perturbed_original_ll"] for r in results], alpha=0.5, bins='auto', label='perturbed original')
            plt.xlabel("log likelihood")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/ll_histograms_{experiment['name']}.png")
        except:
            pass


# save the histograms of log likelihood ratios in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def save_llr_histograms(experiments):
    
    plt.clf()

    for experiment in experiments:
        try:
            results = experiment["raw_results"]
            # plot histogram of sampled/perturbed sampled on left, original/perturbed original on right
            plt.figure(figsize=(20, 6))
            plt.subplot(1, 2, 1)

            # compute the log likelihood ratio for each result
            for r in results:
                r["sampled_llr"] = r["sampled_ll"] - r["perturbed_sampled_ll"]
                r["original_llr"] = r["original_ll"] - r["perturbed_original_ll"]
            
            plt.hist([r["sampled_llr"] for r in results], alpha=0.5, bins='auto', label='sampled')
            plt.hist([r["original_llr"] for r in results], alpha=0.5, bins='auto', label='original')
            plt.xlabel("log likelihood ratio")
            plt.ylabel('count')
            plt.legend(loc='upper right')
            plt.savefig(f"{SAVE_FOLDER}/llr_histograms_{experiment['name']}.png")
        except:
            pass


def get_perturbation_results(span_length=10, n_perturbations=1, n_samples=500):
    load_mask_model()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    original_text = data["original"]
    sampled_text = data["sampled"]

    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=args.pct_words_masked)

    p_sampled_text = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
    for _ in range(n_perturbation_rounds - 1):
        try:
            p_sampled_text, p_original_text = perturb_fn(p_sampled_text), perturb_fn(p_original_text)
        except AssertionError:
            break

    assert len(p_sampled_text) == len(sampled_text) * n_perturbations, f"Expected {len(sampled_text) * n_perturbations} perturbed samples, got {len(p_sampled_text)}"
    assert len(p_original_text) == len(original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

    for idx in range(len(original_text)):
        results.append({
            "original": original_text[idx],
            "sampled": sampled_text[idx],
            "perturbed_sampled": p_sampled_text[idx * n_perturbations: (idx + 1) * n_perturbations],
            "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })

    load_base_model()

    for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
        p_sampled_ll = get_lls(res["perturbed_sampled"])
        p_original_ll = get_lls(res["perturbed_original"])
        res["original_ll"] = get_ll(res["original"])
        res["sampled_ll"] = get_ll(res["sampled"])
        res["all_perturbed_sampled_ll"] = p_sampled_ll
        res["all_perturbed_original_ll"] = p_original_ll
        res["perturbed_sampled_ll"] = np.mean(p_sampled_ll)
        res["perturbed_original_ll"] = np.mean(p_original_ll)
        res["perturbed_sampled_ll_std"] = np.std(p_sampled_ll) if len(p_sampled_ll) > 1 else 1
        res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1

    return results


def run_perturbation_experiment(results, criterion, span_length=10, n_perturbations=1, n_samples=500):
    # compute diffs with perturbed
    predictions = {'real': [], 'samples': []}
    for res in results:
        if criterion == 'd':
            predictions['real'].append(res['original_ll'] - res['perturbed_original_ll'])
            predictions['samples'].append(res['sampled_ll'] - res['perturbed_sampled_ll'])
        elif criterion == 'z':
            if res['perturbed_original_ll_std'] == 0:
                res['perturbed_original_ll_std'] = 1
                print("WARNING: std of perturbed original is 0, setting to 1")
                print(f"Number of unique perturbed original texts: {len(set(res['perturbed_original']))}")
                print(f"Original text: {res['original']}")
            if res['perturbed_sampled_ll_std'] == 0:
                res['perturbed_sampled_ll_std'] = 1
                print("WARNING: std of perturbed sampled is 0, setting to 1")
                print(f"Number of unique perturbed sampled texts: {len(set(res['perturbed_sampled']))}")
                print(f"Sampled text: {res['sampled']}")
            predictions['real'].append((res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])
            predictions['samples'].append((res['sampled_ll'] - res['perturbed_sampled_ll']) / res['perturbed_sampled_ll_std'])

    metric_res = get_overall_metrics(predictions, do_reverse=False)
    name = f'perturbation_{n_perturbations}_{criterion}'
    print(f"{name} ROC AUC: {metric_res['roc_auc']}, PR AUC: {metric_res['pr_auc']} | tpr@fpr=5%: {metric_res['tpr_at_fpr5']} | ACC_95: {metric_res['acc_95']}, ASR_95: {metric_res['asr_95']}, ACC_95_M {metric_res['acc_95_m']} | ACC: {metric_res['acc']}, F1: {metric_res['f1']}, ASR: {metric_res['asr']}")
    return {
        'name': name,
        'predictions': predictions,
        'info': {
            'pct_words_masked': args.pct_words_masked,
            'span_length': span_length,
            'n_perturbations': n_perturbations,
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'acc': metric_res['acc'],
            'f1': metric_res['f1'],
            'asr': metric_res['asr'],
            'acc_95': metric_res['acc_95'],
            'asr_95': metric_res['asr_95'],
            'acc_95_m': metric_res['acc_95_m'],
            'roc_auc': metric_res['roc_auc'],
            'fpr': metric_res['fpr'],
            'tpr': metric_res['tpr'],
            'tpr_at_fpr5': metric_res['tpr_at_fpr5'],
            'tpr_at_fpr10': metric_res['tpr_at_fpr10'],
            'tpr_at_fpr20': metric_res['tpr_at_fpr20'],
        },
        'pr_metrics': {
            'pr_auc': metric_res['pr_auc'],
            'precision': metric_res['p'],
            'recall': metric_res['r'],
        },
        'loss': 1 - metric_res['pr_auc'],
    }


def run_baseline_threshold_experiment(criterion_fn, name, n_samples=500):
    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    for batch in tqdm.tqdm(range(n_samples // batch_size), desc=f"Computing {name} criterion"):
        original_text = data["original"][batch * batch_size:(batch + 1) * batch_size]
        sampled_text = data["sampled"][batch * batch_size:(batch + 1) * batch_size]

        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "original_crit": criterion_fn(original_text[idx]),
                "sampled": sampled_text[idx],
                "sampled_crit": criterion_fn(sampled_text[idx]),
            })

    # compute prediction scores for real/sampled passages
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results],
    }
    metric_res = get_overall_metrics(predictions, do_reverse=False)
    print(f"{name}_threshold ROC AUC: {metric_res['roc_auc']}, PR AUC: {metric_res['pr_auc']} | tpr@fpr=5%: {metric_res['tpr_at_fpr5']} | ACC_95: {metric_res['acc_95']}, ASR_95: {metric_res['asr_95']}, ACC_95_M {metric_res['acc_95_m']} | ACC: {metric_res['acc']}, F1: {metric_res['f1']}, ASR: {metric_res['asr']}")
    return {
        'name': f'{name}_threshold',
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'raw_results': results,
        'metrics': {
            'acc': metric_res['acc'],
            'f1': metric_res['f1'],
            'asr': metric_res['asr'],
            'acc_95': metric_res['acc_95'],
            'asr_95': metric_res['asr_95'],
            'acc_95_m': metric_res['acc_95_m'],
            'roc_auc': metric_res['roc_auc'],
            'fpr': metric_res['fpr'],
            'tpr': metric_res['tpr'],
            'tpr_at_fpr5': metric_res['tpr_at_fpr5'],
            'tpr_at_fpr10': metric_res['tpr_at_fpr10'],
            'tpr_at_fpr20': metric_res['tpr_at_fpr20'],
        },
        'pr_metrics': {
            'pr_auc': metric_res['pr_auc'],
            'precision': metric_res['p'],
            'recall': metric_res['r'],
        },
        'loss': 1 - metric_res['pr_auc'],
    }


# strip newlines from each example; replace one or more newlines with a single space
def strip_newlines(text):
    return ' '.join(text.split())


# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb


def truncate_to_substring(text, substring, idx_occurrence):
    # truncate everything after the idx_occurrence occurrence of substring
    assert idx_occurrence > 0, 'idx_occurrence must be > 0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


def generate_samples(raw_data, batch_size):
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "original": [],
        "sampled": [],
    }

    for batch in range(len(raw_data) // batch_size):
        print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
        original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
        sampled_text = sample_from_model(original_text, min_words=30 if args.dataset in ['pubmed'] else 55)

        for o, s in zip(original_text, sampled_text):
            if args.dataset == 'pubmed':
                s = truncate_to_substring(s, 'Question:', 2)
                o = o.replace(custom_datasets.SEPARATOR, ' ')

            o, s = trim_to_shorter_length(o, s)

            # add to the data
            data["original"].append(o)
            data["sampled"].append(s)
    
    if args.pre_perturb_pct > 0:
        print(f'APPLYING {args.pre_perturb_pct}, {args.pre_perturb_span_length} PRE-PERTURBATIONS')
        load_mask_model()
        data["sampled"] = perturb_texts(data["sampled"], args.pre_perturb_span_length, args.pre_perturb_pct, ceil_pct=True)
        load_base_model()

    return data


def generate_data(dataset, key):
    # load data
    if dataset in custom_datasets.DATASETS:
        data = custom_datasets.load(dataset, cache_dir)
    else:
        data = datasets.load_dataset(dataset, split='train', cache_dir=cache_dir)[key]

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the mask model)
    # then generate n_samples samples

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [x.strip() for x in data]

    # remove newlines from each example
    data = [strip_newlines(x) for x in data]

    # try to keep only examples with > 250 words
    if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 250]
        if len(long_data) > 0:
            data = long_data

    random.seed(0)
    random.shuffle(data)

    data = data[:5_000]

    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

    # print stats about remainining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return generate_samples(data[:n_samples], batch_size=batch_size)


def load_base_model_and_tokenizer(name):
    if args.openai_model is None:
        print(f'Loading BASE model {args.base_model_name}...')
        base_model_kwargs = {}
        if 'gpt-j' in name or 'neox' in name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in name:
            base_model_kwargs.update(dict(revision='float16'))
        if name=='gpt-j':
            name='EleutherAI/gpt-j-6b'
        base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=cache_dir)
    else:
        base_model = None

    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if args.dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer


def eval_supervised(data, model):
    print(f'Beginning supervised evaluation with {model}...')
    if model.split('.')[-1] == 'pt':
        print('Loading local model...')
        detector = torch.load(model).to(DEVICE)
        tokenizer = transformers.AutoTokenizer.from_pretrained('microsoft/deberta-v3-base',cache_dir=cache_dir)
    else:
        detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir=cache_dir).to(DEVICE)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)

    real, fake = data['original'], data['sampled']

    with torch.no_grad():
        # get predictions for real
        real_preds = []
        for batch in tqdm.tqdm(range(len(real) // batch_size), desc="Evaluating real"):
            batch_real = real[batch * batch_size:(batch + 1) * batch_size]
            batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            real_preds.extend(detector(**batch_real).logits.softmax(-1)[:,0].tolist())
        
        # get predictions for fake
        fake_preds = []
        for batch in tqdm.tqdm(range(len(fake) // batch_size), desc="Evaluating fake"):
            batch_fake = fake[batch * batch_size:(batch + 1) * batch_size]
            batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            fake_preds.extend(detector(**batch_fake).logits.softmax(-1)[:,0].tolist())

    predictions = {
        'real': real_preds,
        'samples': fake_preds,
    }
    if model.split('.')[-1] == 'pt':
        # since the label numbers are in adverse.
        predictions = {
            'samples': real_preds,
            'real': fake_preds,
        }
    metric_res = get_overall_metrics(predictions, do_reverse=False)
    print(f"{model} ROC AUC: {metric_res['roc_auc']}, PR AUC: {metric_res['pr_auc']} | tpr@fpr=5%: {metric_res['tpr_at_fpr5']} | ACC_95: {metric_res['acc_95']}, ASR_95: {metric_res['asr_95']}, ACC_95_M {metric_res['acc_95_m']} | ACC: {metric_res['acc']}, F1: {metric_res['f1']}, ASR: {metric_res['asr']}")

    # free GPU memory
    del detector
    torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'metrics': {
            'acc': metric_res['acc'],
            'f1': metric_res['f1'],
            'asr': metric_res['asr'],
            'acc_95': metric_res['acc_95'],
            'asr_95': metric_res['asr_95'],
            'acc_95_m': metric_res['acc_95_m'],
            'roc_auc': metric_res['roc_auc'],
            'fpr': metric_res['fpr'],
            'tpr': metric_res['tpr'],
            'tpr_at_fpr5': metric_res['tpr_at_fpr5'],
            'tpr_at_fpr10': metric_res['tpr_at_fpr10'],
            'tpr_at_fpr20': metric_res['tpr_at_fpr20'],
        },
        'pr_metrics': {
            'pr_auc': metric_res['pr_auc'],
            'precision': metric_res['p'],
            'recall': metric_res['r'],
        },
        'loss': 1 - metric_res['pr_auc'],
    }

def load_custom_data(args):
    generation_model = args.generate_model_name
    if args.do_attack!="False":
        dataset_att_path = f"{args.dataset}_model_data/{args.dataset_key}/{generation_model}_test.{args.attack_method+args.attack_args}_att.csv"
        dataset_ori_path = f"{args.dataset}_model_data/{args.dataset_key}/{generation_model}_test.csv"
        test_dataset_ori = load_csv(dataset_ori_path)
        test_dataset_att = load_csv(dataset_att_path)
        test_dataset_att_mgt = [d for d in test_dataset_att if d['label']=='0']
        test_dataset_ori_hwt = [d for d in test_dataset_ori if d['label']=='1']
        data = test_dataset_att_mgt + test_dataset_ori_hwt
    else:
        dataset_path = f"{args.dataset}_model_data/{args.dataset_key}/{generation_model}_test.csv"
        data = load_csv(dataset_path)

    random.seed(0)
    random.shuffle(data)

    data = data

    labels = [d['label'] for d in data]
    texts = [d['sequence'] for d in data]
    data = texts
    # keep only examples with <= 512 tokens according to mask_tokenizer
    # this step has the extra effect of removing examples with low-quality/garbage content
    # tokenized_data = preproc_tokenizer(data)
    # data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

    # print stats about remainining data
    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    output_dict = {'original': [t for t,l in zip(texts, labels) if l=='1'][:args.n_samples], 'sampled': [t for t,l in zip(texts, labels) if l=='0'][:args.n_samples]}
    return output_dict

if __name__ == '__main__':
    DEVICE = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="multi_model_data")
    parser.add_argument('--dataset_key', type=str, default="news_gptj_t1.5")
    parser.add_argument('--pct_words_masked', 
                        type=float, 
                        default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=500, help="number of dataset examples (per label)")
    parser.add_argument('--n_perturbation_list', type=str, default="1,10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--base_model_name', type=str, default="gpt-j")
    parser.add_argument('--generate_model_name', type=str, default="gptj")
    parser.add_argument('--scoring_model_name', type=str, default="")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-3b")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=5)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--output_name', type=str, default="main")
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--baselines_only', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="/home/gridsan/tianxing/txml_shared/yichen/cache")
    parser.add_argument('--gpu_id', 
                        type=str, default="0")
    
    parser.add_argument('--do_attack', 
                        type=str, default=True)
    parser.add_argument('--attack_method', 
                        type=str, default='typo')
    parser.add_argument('--attack_args', 
                        type=str, default='0.3')
    parser.add_argument('--output_suffix',
                        type=str, default='patched')
    args = parser.parse_args()

    API_TOKEN_COUNTER = 0

    if args.openai_model is not None:
        import openai
        assert args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
        openai.api_key = args.openai_key

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')
    print("***Currently Running run_WITH_FILTER.py ***")
    print(START_DATE, START_TIME)
    print(args)
    # define SAVE_FOLDER as the timestamp - base model name - mask filling model name
    # create it if it doesn't exist
    precision_string = "int8" if args.int8 else ("fp16" if args.half else "fp32")
    sampling_string = "top_k" if args.do_top_k else ("top_p" if args.do_top_p else "temp")
    output_subfolder = f"{args.output_name}/" if args.output_name else ""
    if args.openai_model is None:
        base_model_name = args.base_model_name.replace('/', '_')
    else:
        base_model_name = "openai-" + args.openai_model.replace('/', '_')
    scoring_model_string = (f"-{args.scoring_model_name}" if args.scoring_model_name else "").replace('/', '_')
    SAVE_FOLDER = f"tmp_results/{output_subfolder}{base_model_name}{scoring_model_string}-{args.mask_filling_model_name}-{sampling_string}/{START_DATE}-{START_TIME}-{precision_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{args.dataset}-{args.n_samples}{args.output_suffix}"
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

    XLSX_PATH = f"logs/{args.generate_model_name}/{args.generate_model_name}@{args.base_model_name.replace('/', '-')}.{args.output_suffix}_{args.attack_method}{args.attack_args if args.attack_args!='' else 'None'}.xlsx"
    if not os.path.exists(f"logs/{args.generate_model_name}"):
        os.makedirs(f"logs/{args.generate_model_name}")
    print(f"Saving Excel as absolute path: {os.path.abspath(XLSX_PATH)}")

    # write args to file
    with open(os.path.join(SAVE_FOLDER, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

    # generic generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name)

    # mask filling t5 model
    if not args.baselines_only and not args.random_fills:
        int8_kwargs = {}
        half_kwargs = {}
        if args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir=cache_dir)
        try:
            n_positions = mask_model.config.n_positions
        except AttributeError:
            n_positions = 512
    else:
        n_positions = 512
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=cache_dir)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions, cache_dir=cache_dir)
    if args.dataset in ['english', 'german']:
        preproc_tokenizer = mask_tokenizer

    load_base_model()

    print(f'Loading dataset {args.dataset}...')
    if args.dataset == "multi":
        data = load_custom_data(args)
    else:
        data = generate_data(args.dataset, args.dataset_key) # data {'original': [], 'sampled': []} len 500
        random.shuffle(data['sampled'])
    for idx, text in enumerate(data['sampled']):
        if text == '':
            print("Existing blank data.")
            data['sampled'][idx] = '<blank text>'
    if args.random_fills:
        FILL_DICTIONARY = set()
        for texts in data.values():
            for text in texts:
                FILL_DICTIONARY.update(text.split())
        FILL_DICTIONARY = sorted(list(FILL_DICTIONARY))

    if args.scoring_model_name:
        print(f'Loading SCORING model {args.scoring_model_name}...')
        del base_model
        del base_tokenizer
        torch.cuda.empty_cache()
        base_model, base_tokenizer = load_base_model_and_tokenizer(args.scoring_model_name)
        load_base_model()  # Load again because we've deleted/replaced the old model

    # write the data to a json file in the save folder
    with open(os.path.join(SAVE_FOLDER, "raw_data.json"), "w") as f:
        print(f"Writing raw data to {os.path.join(SAVE_FOLDER, 'raw_data.json')}")
        json.dump(data, f)

    if not args.skip_baselines:
        baseline_outputs = [run_baseline_threshold_experiment(get_ll, "likelihood", n_samples=n_samples)]
        if args.openai_model is None:
            rank_criterion = lambda text: -get_rank(text, log=False)
            baseline_outputs.append(run_baseline_threshold_experiment(rank_criterion, "rank", n_samples=n_samples))
            logrank_criterion = lambda text: -get_rank(text, log=True)
            baseline_outputs.append(run_baseline_threshold_experiment(logrank_criterion, "log_rank", n_samples=n_samples))
            entropy_criterion = lambda text: get_entropy(text)
            baseline_outputs.append(run_baseline_threshold_experiment(entropy_criterion, "entropy", n_samples=n_samples))

        baseline_outputs.append(eval_supervised(data, model='roberta-base-openai-detector'))
        baseline_outputs.append(eval_supervised(data, model='roberta-large-openai-detector'))
        baseline_outputs.append(eval_supervised(data, model='Hello-SimpleAI/chatgpt-detector-roberta'))
        baseline_outputs.append(eval_supervised(data, model=f'/home/gridsan/tianxing/txml_shared/yichen/Robust2Att-Bkl2/models/microsoft_deberta-v3-large@D_{args.generate_model_name}@G.pt'))

    outputs = []

    if not args.baselines_only:
        # run perturbation experiments
        for n_perturbations in n_perturbation_list:
            perturbation_results = get_perturbation_results(args.span_length, n_perturbations, n_samples)
            for perturbation_mode in ['d', 'z']:
                output = run_perturbation_experiment(
                    perturbation_results, perturbation_mode, span_length=args.span_length, n_perturbations=n_perturbations, n_samples=n_samples)
                outputs.append(output)
                with open(os.path.join(SAVE_FOLDER, f"perturbation_{n_perturbations}_{perturbation_mode}_results.json"), "w") as f:
                    json.dump(output, f)

    if not args.skip_baselines:
        # write likelihood threshold results to a file
        with open(os.path.join(SAVE_FOLDER, f"likelihood_threshold_results.json"), "w") as f:
            json.dump(baseline_outputs[0], f)

        if args.openai_model is None:
            # write rank threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"rank_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[1], f)

            # write log rank threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"logrank_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[2], f)

            # write entropy threshold results to a file
            with open(os.path.join(SAVE_FOLDER, f"entropy_threshold_results.json"), "w") as f:
                json.dump(baseline_outputs[3], f)
        
        # write supervised results to a file
        with open(os.path.join(SAVE_FOLDER, f"roberta-base-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-3], f)
        
        # write supervised results to a file
        with open(os.path.join(SAVE_FOLDER, f"roberta-large-openai-detector_results.json"), "w") as f:
            json.dump(baseline_outputs[-2], f)

        with open(os.path.join(SAVE_FOLDER, f"chatgpt-detector-roberta.json"), "w") as f:
            json.dump(baseline_outputs[-1], f)

        outputs += baseline_outputs


    # save to xlsx
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    xlsx_data = [["Method", "ROC AUC", "PR AUC", "ASR", "tpr@fpr=20%", "tpr@fpr=10%", "tpr@fpr=5%", "ACC_95", "ASR_95", "ACC_95_M", "ACC", "F1"]]

    xlse_outputs = outputs[4:8] + outputs[0:1] + outputs[2:4] + outputs[8:]
    for output in xlse_outputs:
        metrics = output["metrics"]
        pr_metrics = output['pr_metrics']

        xlsx_data.append([output['name'], f"{metrics['roc_auc']:.5f}", f"{pr_metrics['pr_auc']:.5f}", f"{metrics['asr']:.5f}", f"{metrics['tpr_at_fpr20']:.5f}", f"{metrics['tpr_at_fpr10']:.5f}", f"{metrics['tpr_at_fpr5']:.5f}", f"{metrics['acc_95']:.5f}", f"{metrics['asr_95']:.5f}", f"{metrics['acc_95_m']:.5f}",  f"{metrics['acc']:.5f}", f"{metrics['f1']:.5f}"])

    for row in xlsx_data:
        sheet.append(row)
    # Save the workbook
    workbook.save(XLSX_PATH)
    save_roc_curves(outputs)
    save_ll_histograms(outputs)
    save_llr_histograms(outputs)

    # move results folder from tmp_results/ to results/, making sure necessary directories exist
    new_folder = SAVE_FOLDER.replace("tmp_results", "results")
    if not os.path.exists(os.path.dirname(new_folder)):
        os.makedirs(os.path.dirname(new_folder))
    os.rename(SAVE_FOLDER, new_folder)

    print(f"Used an *estimated* {API_TOKEN_COUNTER} API tokens (may be inaccurate)")