import os
import torch
from tqdm import tqdm
import pandas as pd
import argparse
import nltk
from openai import OpenAI
from utils_gen import get_prompt, count_tokens, truncate

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
        default="gpt-4",
        help="official name for model / for saving dataset",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="multi_model_data",
        help="csv file including positive samples",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7, 
        help="",
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

def gpt4_completion(prompt):
    args = parse_args()
    client = OpenAI(
    api_key="your/openai/key",
    )
    messages = [{"role": "user", "content": "Please continue this text in about 90 words:" + prompt.strip()}]
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4",
        top_p=args.top_p,
        temperature=args.temp,
        max_tokens=160
    )
    ans = response.choices[0].message.content
    return ans

def main():
    args = parse_args()
    split = 'test'

    prompt_path = os.path.join(args.input_path, args.domain) + "/" + args.load_dataset_model +f"_{split}_{args.paraphraser}_para_prompt.csv"
    out_path = os.path.join(args.input_path, args.domain) + "/" + args.gen_model_name +f"_{split}_{args.paraphraser}_para_prompt_t{args.temp}.csv"

    df = pd.read_csv(prompt_path, sep="|")
    print("*Prompts loaded from", prompt_path)
    print("*Output dir", out_path)

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
    else:
        args.model = model_name

    try_times = MAX_TRIAL
    prompts = []
    pr_index = []
    rep_tot, gen_tot = 0,0
    from multi_model_data.repeating_detect import evaluate_text
    for index, d in tqdm(df.iterrows()):
        if d["label"] == 1: # human written texts
            continue
        else: # machine generated texts
            seq = d["sequence"]
            try:
                for idx in range(try_times):
                    prompt = d["prompt"]
                    if model_name not in ["text-davinci-003","gpt-4"]:
                        raise NotImplementedError
                    else:
                        if model_name == "gpt-4":
                            ans = gpt4_completion(prompt)
                        else:
                            ans = davinci_completion(model_name, prompt)

                        if args.model == "gpt-4":
                            decoded_output = exp_truncate(prompt.strip() + " " + ans, 110)
                        else:
                            decoded_output = truncate(prompt.strip() + " " + ans,110)
                        token_num = count_tokens(decoded_output)
                        if token_num in range(100, 121):
                            df.at[index, "sequence"] = decoded_output
                            break
                        if idx == try_times-1:
                            print(token_num)
                            df.at[index, "sequence"] = "<blank text>"

            except RuntimeError:
                print("Runtime Error!")  
        gen_tot += 1
        rep = evaluate_text(decoded_output)
        if rep:
            print(f"***Repeating {rep} times***")
            rep_tot += rep
            print(f"===Repeating tot {rep_tot/gen_tot}={rep_tot}/{gen_tot} times===")

    df.to_csv(out_path, sep = "|", index = None)
    print("Writing csv file to " + out_path)

if __name__ == "__main__":
    main()
