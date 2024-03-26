import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
import argparse
from utils_gen import get_prompt, count_tokens,truncate
import os
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
import openai
import nltk
import random

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--domain",
        type=str,
        default="news",
    )
    parser.add_argument(
        "--load_dataset_model",
        type=str,
        default="gptj",
    )
    parser.add_argument(
        "--gen_model_name", 
        type=str,
        default="gptj",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="multi_model_data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
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
        "--gpu_id",
        type=str,
        default="0",
    )
    args = parser.parse_args()
    return args
    
def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

seed_everything(1)

def exp_truncate(text,tgt):
    now = 0
    sens = nltk.sent_tokenize(text)
    for i in range(len(sens)):
        if now < tgt:
            last = now
            now += count_tokens(sens[i])
        else:
            break
    if tgt-last < now-tgt:
        out_sens = i
    else:
        out_sens = i+1
    res = " ".join(sens[:out_sens])
    return res


MAX_TRIAL = 10

def get_prompt(text, prompt_len=20):
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens[:prompt_len])

def main(): 
    args = parse_args()
    split = 'val'
    out_path = os.path.join(args.input_path, args.domain) + "/" + args.gen_model_name +f"_{split}_prompt.csv"

    print("*Output to file:", out_path)
  
    TESTSET_PATH = os.path.join(args.input_path, args.domain + "/" + args.load_dataset_model + f"_{split}.csv")

    df = pd.read_csv(TESTSET_PATH, sep="|")
    print("*Loaded from", TESTSET_PATH)

    model_name = args.gen_model_name
    if model_name in ["openai-gpt","gpt2","gpt2-medium","gpt2-large","gpt2-xl"]:
        args.model = model_name
    elif model_name == "gptneo-sm":
        args.model = "EleutherAI/gpt-neo-125m"
    elif model_name == "gptneo-md":
        args.model = "EleutherAI/gpt-neo-1.3B"
    elif model_name == "gptneo-lg":
        args.model = "EleutherAI/gpt-neo-2.7B"
    elif model_name == "gptj":
        args.model = "EleutherAI/gpt-j-6b"
    elif model_name[:3] == "opt":
        args.model = "facebook/" + model_name
    elif model_name[:6] == "pythia":
        args.model = "EleutherAI/" + model_name
    elif model_name[:6] == "bloomz":
        args.model = "bigscience/" + model_name
    elif model_name == "Llama-2-7b-hf":
        args.model = "meta-llama/Llama-2-7b-hf"
    elif model_name in ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-4"]:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        args.model = model_name

    for index, d in tqdm(df.iterrows()):
        if d["label"] == 1: # human written texts
            df.at[index, "prompt"] = ""
            continue
        seq = d["sequence"]
        prompt = get_prompt(seq)
        df.at[index, "prompt"] = prompt 
    df.to_csv(out_path, sep = "|", index = None)
    print("Writing csv file to " + out_path)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
