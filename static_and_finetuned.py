import os
import torch

torch.cuda.empty_cache() 
import argparse
import random
import numpy as np
from utils import load_csv, number_h, compute_metrics, histogram_word, to_tensor_dataset, get_overall_metrics
from torch.utils.data import (
    DataLoader,
    DataLoader,
    SequentialSampler,
)
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
)
from distance_measure import dist_and_sim

CACHE_DIR = "/cache"

TASK = "static"

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ind_model_name", 
        type=str, 
        default="gptj",
        help="The dataset that model trained on. For select the fine-tuned models."
    )
    parser.add_argument(
        "--base_model_name",
        default="microsoft/deberta-v3-large",  
        type=str,
        help="The structure of detector.",
    )
    parser.add_argument(
        "--testset_ori",
        default= "multi_model_data/news_gptj_t1.5/gptj_test.csv", 
        type=str, 
    )
    parser.add_argument(
        "--testset_att",
        default= "multi_model_data/news_gptj_t1.5/gptj_test.word_subst_modelfree0.02_att.csv", 
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.getcwd(), "results/cls"),
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--seed", type=int, default=82, help="random seed for initialization"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help=""
    )
    parser.add_argument(
        "--watermark", action='store_true')
    
    parser.add_argument(
        "--not_include_model_based", action='store_false')
    
    args = parser.parse_args()
    return args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def _eval(args, eval_dataset, model, tokenizer, mode, epoch=None):
    loss_fn = torch.nn.CrossEntropyLoss()
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    )
    eval_loss, eval_step = 0.0, 0
    preds = None
    with logging_redirect_tqdm():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[1], "attention_mask": batch[2]}
                outputs = model(**inputs)
                
                scores_softmax = F.softmax(outputs.logits, dim=1)[:, 0].squeeze(-1)
                scores = outputs.logits[:, 0].squeeze(-1)
                loss = loss_fn(scores, batch[0])
                
            eval_step += 1
            if preds is None:
                preds = scores.detach().cpu().numpy()
                preds_softmax = scores_softmax.detach().cpu().numpy()
                labels = batch[0].detach().cpu().numpy()
            else:
                preds = np.append(preds, scores.detach().cpu().numpy(), axis=0)
                preds_softmax = np.append(preds_softmax, scores_softmax.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, batch[0].detach().cpu().numpy(), axis=0)
   
    preds = preds.reshape(-1)
    preds_softmax = preds_softmax.reshape(-1)
    histogram_word(preds_softmax)

    preds_softmax_clsed = {
        'real': [p for p,l in zip(preds_softmax, labels) if l==0],
        'samples': [p for p,l in zip(preds_softmax, labels) if l==1]
    }
    result = get_overall_metrics(preds_softmax_clsed, do_reverse=False)

    return result, loss


def cls_eval(args, data=None):

    print("*** Attack on Classifier ***")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current Device: %s", args.device)

    MODEL_PATH = f"models/{args.base_model_name.replace('/', '_')}@D_{args.ind_model_name}@G.pt"
    print(f"* Model: {args.base_model_name.replace('/', '_')}@D_{args.ind_model_name}@G ***")


    torch.cuda.empty_cache()
    if args.base_model_name == "microsoft/deberta-v3-base":
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name,)
    model = torch.load(MODEL_PATH).to(args.device)
    if args.base_model_name == "openai-gpt":
        tokenizer.pad_token = "pad_token"
        model.config.pad_token_id = 0
    elif tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    if data is not None: 
        test_dataset  = to_tensor_dataset(args, data, tokenizer)
        result, _ = _eval(args, test_dataset, model, tokenizer, mode="test")
        for key in result.keys():
            if type(result[key]) is not list:
                print(key, "=", f"{result[key]:.5f}")
        return result
    else: 
        raise NotImplementedError
      
def clean_texts(texts):
    for t in texts:
        t["sequence"] = clean_text_(t["sequence"])
    return texts

def clean_text_(text):
    allowed_special_chars = set(",.!?'; \u200B\u000B")
    cleaned_text = ''.join(char for char in text if char.isalnum() or char in allowed_special_chars)
    cleaned_text = ''.join(char if char.isalnum() or char == ' ' else f' {char} ' for char in cleaned_text)
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

def get_key(para, length=8):
    """Returns the first 'length' characters without spaces."""
    return ''.join(para.split()).replace('"', '').replace('`', '')[:length]

def main():
    args = init_args()
    set_seed(args)
    print(args)
    if TASK == "static":
        if args.watermark:
            XLSX_PATH = f"logs/{args.ind_model_name}watermark/{args.ind_model_name}_stat_{'.'.join(args.testset_att.split('/')[-1].split('.')[1:-1])}.xlsx"
        else:
            XLSX_PATH = f"logs/{args.ind_model_name}/{args.ind_model_name}_stat_{'.'.join(args.testset_att.split('/')[-1].split('.')[1:-1])}.xlsx"
        print(f"* Xlsx Output: {XLSX_PATH}")
    elif TASK == "finetune":
        XLSX_PATH = f"logs/gptj/gptj@gpt-j_{'.'.join(args.testset_att.split('/')[-1].split('.')[1:-1]).split('_')[0]}.xlsx"
        print(f"Saving Excel as absolute path: {os.path.abspath(XLSX_PATH)}") 
    test_dataset_ori = load_csv(args.testset_ori)
    test_dataset_att = load_csv(args.testset_att)
    
    test_dataset_att_mgt = [d for d in test_dataset_att if d['label']=='0']
    test_dataset_ori_mgt = [d for d in test_dataset_ori if d['label']=='0']
    test_dataset_ori_hwt = [d for d in test_dataset_ori if d['label']=='1']
    test_dataset_att_mgt = clean_texts(test_dataset_att_mgt)
    test_dataset_ori_mgt = clean_texts(test_dataset_ori_mgt)
    test_dataset_ori_hwt = clean_texts(test_dataset_ori_hwt) 

    test_dataset_att = test_dataset_att_mgt + test_dataset_ori_hwt
    
    matching_file = "multi_model_data/news_gptj_t1.5/id_matching.json"
    import json
    if os.path.exists(matching_file):
        print("*loading matches*")
        ordered_test_dataset_ori_hwt = []
        with open(matching_file, 'r') as file:
            matching_hwt_and_mgt = json.load(file)
        for idx_pair in matching_hwt_and_mgt:
            ordered_test_dataset_ori_hwt.append(test_dataset_ori_hwt[idx_pair["hwt_id"]])
    else:
        print("*not found matches cache file, computing*")
        dict2 = {get_key(para['sequence']): (hwt_id, para) for hwt_id,para in enumerate(test_dataset_ori_hwt)}
        ordered_test_dataset_ori_hwt = []
        matching_hwt_and_mgt = []
        for mgt_id, para in enumerate(test_dataset_ori_mgt):
            key = get_key(para['sequence'])
            if key in dict2.keys():
                ordered_test_dataset_ori_hwt.append(dict2[key][1])
                matching_hwt_and_mgt.append({"mgt_id": mgt_id, "hwt_id":dict2[key][0]})
            else:
                print(key)
                ordered_test_dataset_ori_hwt.append(key)

    if TASK == "static":      
        xlsx_data = dist_and_sim(test_dataset_ori_mgt, test_dataset_att_mgt, ordered_test_dataset_ori_hwt, do_model_based=args.not_include_model_based)

        import openpyxl
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        for row in xlsx_data:
            sheet.append(row)

        workbook.save(XLSX_PATH)
    
if __name__ == "__main__":
    main()