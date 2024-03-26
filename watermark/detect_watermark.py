import os
import torch

from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

random.seed(0)
import argparse
import openpyxl
from detect.run import get_overall_metrics
from utils import load_csv

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--domain",
        type=str,
        default="news_gptj_watermark",
        help="choose from 'news','review' and 'wiki'",
    )
    parser.add_argument(
        "--gen_model_name",
        type=str,
        default="gptj",
        help="",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="multi_model_data",
        help="csv file including positive samples",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/cache",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.96,
        help="",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.5,
        help="",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="",
    )
    parser.add_argument(
        "--rp",
        type=float,
        default=1.0,
        help="",
    )
    parser.add_argument(
        "--do_sample",
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        help="",
    )
    parser.add_argument(
        "--attack",
        type=str,
        help="",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.base_model_name = args.gen_model_name

    args.device = torch.device("cuda" if torch.cuda.is_available() else ValueError)
    cuda_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = torch.cuda.memory_allocated(0)
    free_memory = cuda_memory - available_memory

    print(f"{free_memory} = {cuda_memory} - {available_memory}")

    TESTSET_PATH = os.path.join(
        args.input_path,
        args.domain + "/" + args.gen_model_name + f"_test.{args.attack}.csv",
    )
    XLSX_PATH = (
        f"logs/{args.gen_model_name}watermark/{args.attack}_{args.base_model_name.replace('/', '-')}.xlsx"
    )
    print("Saving to xlsx in", XLSX_PATH)

    df_mgt = load_csv(TESTSET_PATH)
    df_mgt = [d for d in df_mgt if d['label']=='0']
    df_hwt = load_csv('multi_model_data/news_gptj_watermark/gptj_test.csv')
    df_hwt = [d for d in df_hwt if d['label']=='1']
    df = df_mgt + df_hwt

    model_name = args.gen_model_name
    if model_name in ["openai-gpt", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
        args.model = model_name
    elif model_name == "gptneo-sm":
        args.model = "EleutherAI/gpt-neo-125m"
    elif model_name == "gptneo-md":
        args.model = "EleutherAI/gpt-neo-1.3B"
    elif model_name == "gptneo-lg":
        args.model = "EleutherAI/gpt-neo-2.7B"
    elif model_name == "gptj":
        args.model = "EleutherAI/gpt-j-6b"

    if model_name in ["text-davinci-003", "gpt-4"]:
        print("generating samples with " + model_name)
    else:
        print("loading tokenizer " + args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
        print("loading model " + args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir)
        print("moving to gpu")
        if model_name == "llama13B":
            model = model.to(torch.bfloat16).cuda()
        else:
            model = model.cuda()
    print("finish loading")

    from extended_watermark_processor import WatermarkDetector
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.25,  # should match original setting
        seeding_scheme="selfhash",  # should match original setting
        device=model.device,  # must match the original rng device type
        tokenizer=tokenizer,
        z_threshold=4.0,
        normalizers=[],
        ignore_repeated_ngrams=True,
    )
    labels, preds = [], []
    for index, d in enumerate(tqdm(df)):
        try:
            score_dict = watermark_detector.detect(d["sequence"])
        except:
            print(f"Index {index} Fails, SKIP")
            continue
        labels.append(d["label"])
        preds.append(score_dict["z_score"])
    predictions = {"real": [], "samples": []}
    for l, p in zip(labels, preds):
        if l == '1':
            predictions["real"].append(p)
        else:
            predictions["samples"].append(p)
    output = get_overall_metrics(predictions, True)

    # save result to Excel
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    xlsx_data = [["Method", "ROC AUC", "PR AUC", "ASR", "tpr@fpr=20%", "tpr@fpr=10%", "tpr@fpr=5%", "ACC_95", "ASR_95", "ACC_95_M", "ACC", "F1"]]
    xlsx_data.append(["Watermark", f"{output['roc_auc']:.5f}", f"{output['pr_auc']:.5f}", f"{output['asr']:.5f}", f"{output['tpr_at_fpr20']:.5f}", f"{output['tpr_at_fpr10']:.5f}", f"{output['tpr_at_fpr5']:.5f}", f"{output['acc_95']:.5f}", f"{output['asr_95']:.5f}", f"{output['acc_95_m']:.5f}",  f"{output['acc']:.5f}", f"{output['f1']:.5f}"])
    for row in xlsx_data:
        sheet.append(row)

    workbook.save(XLSX_PATH)
    print("Saved to", XLSX_PATH)


if __name__ == "__main__":
    main()
