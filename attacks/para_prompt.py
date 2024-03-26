import os
import torch
from tqdm import tqdm
import pandas as pd
import argparse
from paraphrase import paraphrase

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--domain",
        type=str,
        default="news",
        help="choose from 'news','review' and 'wiki'",
    )
    parser.add_argument(
        "--load_dataset_model",
        type=str,
        default="gptj",
        help="for loading dataset, match with file name.",
    )
    parser.add_argument(
        "--gen_model_name", 
        type=str,
        default="gptj",
        help="official name for model / for saving dataset",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="multi_model_data",
        help="csv file including positive samples",
    )
    parser.add_argument(
        "--paraphraser",
        type=str,
        default="pegasus",
        help="",
    )  
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.96,
        help="",
    )  
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
    )
    args = parser.parse_args()
    return args

MAX_TRIAL = 10
def main():
    args = parse_args()
    split = 'test'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"Current Visible GPU: {args.gpu_id}")
    args.device = torch.device("cuda" if torch.cuda.is_available() else ValueError)
    print("Current Device:", args.device)

    prompt_path = os.path.join(args.input_path, args.domain) + "/" + args.gen_model_name + f"_{split}_prompt.csv"
    out_path = os.path.join(args.input_path, args.domain) + "/" + args.gen_model_name + f"_{split}_{args.paraphraser}_para_prompt.csv"

    df = pd.read_csv(prompt_path, sep="|")
    print("*Loaded from", prompt_path)

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
    elif model_name[:3] == "opt":
        args.model = "facebook/" + model_name
    elif model_name[:6] == "pythia":
        args.model = "EleutherAI/" + model_name
    elif model_name[:6] == "bloomz":
        args.model = "bigscience/" + model_name
    elif model_name == "Llama-2-7b-hf":
        args.model = "meta-llama/Llama-2-7b-hf"
    else:
        args.model = model_name

    try_times = MAX_TRIAL
    prompts = []
    pr_index = []
    for index, d in tqdm(df.iterrows()):
        if d["label"] == 1: # human written texts
            continue
        else: # machine generated texts
            prompts.append(d["prompt"])
            pr_index.append(index)

    args.attack_method = args.paraphraser
    print("Paraprasing ...")
    pr_prompts = paraphrase(args, prompts)
    print("Finalized.")

    for ind, pr_prompt in zip(pr_index, pr_prompts):
        if pr_prompt[-1] == '.':
            pr_prompt = pr_prompt[:-1]
        df.at[ind, "prompt"] = pr_prompt

    df.to_csv(out_path, sep = "|", index = None)
    print("Writing csv file to " + out_path)

if __name__ == "__main__":
    main()
