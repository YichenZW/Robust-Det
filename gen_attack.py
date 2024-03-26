import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
import torch.nn.functional as nn
import argparse
from utils_gen import get_prompt, count_tokens, truncate
import os
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
import openai
import nltk
import random
from attacks.paraphrase import paraphrase
import numpy as np
import string
import copy
import re
from openai import OpenAI
from emoji_loader import emoji_list
CACHE_DIR="/cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:40960"
MAX_TRIAL = 10

SUBST_RULE = {"c":"k", "k":"c"}

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--attack_method",
        type=str,
        default="typo-cogen",
        help="choose from baseline, emoji, icl, csgen, typo-cogen",
    )
    parser.add_argument(
        "--attack_args",
        type=str,
        default="0.75",
        help="",
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        default="news_gptj_t1.5", 
        help="choose from 'news','review' and 'wiki', and name with format like 'news_gpt-4_t0.7'",
    )
    parser.add_argument(
        "--model_list",
        type=str,
        default="EleutherAI/gpt-j-6b",
        help="Use official model name, e.g., gpt-4, EleutherAI/gpt-j-6b",
    )
    parser.add_argument(
        "--gen_model_name",
        type=str,
        default="gptj",
        help="Use unofficial model name, for load dataset, e.g., gpt-4, gptj, ... (since '/' can not be in the name of dataset file)",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="multi_model_data",
        help="csv file including positive samples",
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

def gpt4_completion(prompt):
    args = parse_args()
    client = OpenAI(
    api_key="your/openai/key",
    )
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4",
        top_p=args.top_p,
        temperature=args.temp,
        max_tokens=130 + count_tokens(prompt)
    )
    ans = response.choices[0].message.content
    return ans

def get_prompt(text, prompt_len = 20):
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens[:prompt_len])

def baseline(args, prompt, tokenizer, model):
    try_times = MAX_TRIAL
    encoded_input = tokenizer(prompt, return_tensors='pt').to("cuda")
    for idx in range(try_times):
        output = model.generate(encoded_input.input_ids,
                                top_p=args.top_p,
                                top_k=None, 
                                temperature=args.temp,
                                repetition_penalty=args.rp,
                                num_beams=args.num_beams,
                                do_sample=args.do_sample,
                                min_length=120,
                                max_length=500,
                                pad_token_id=tokenizer.eos_token_id
                                )
        decoded_output = tokenizer.batch_decode(output,
                                                skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)[0]
        decoded_output = truncate(decoded_output, 110)
        token_num = count_tokens(decoded_output)
        if token_num in range(100, 121):    
            return decoded_output
        if idx == try_times - 1:
            print("*Fails", token_num)
            return "<blank text>"

MAX_LENGTH = 160
EMOJIS = emoji_list().list
def generate_with_emojis(args, prompt, tokenizer, model, prob):
    encoded_input = tokenizer(prompt, return_tensors='pt').to("cuda")
    input_ids = copy.deepcopy(encoded_input.input_ids).cpu()
    clean_ids = input_ids
    emoji_num = 0
    while len(input_ids[0]) < MAX_LENGTH:
        with torch.no_grad():
            outputs = model(input_ids.cuda().int())
            next_token_logits = outputs.logits[:, -1, :].cpu()
            # Softmax and sample a token
            probs = nn.softmax(next_token_logits, dim=1)
            total = sum(probs[0].tolist())
            normalized_probs = [x/total for x in probs[0].tolist()]
            next_token = np.random.choice(len(probs[0]), p=normalized_probs)
            if next_token == tokenizer.eos_token_id:
                continue
        # If the generated token is a word (you'll need a way to check this), append an emoji
        # This is a simplified check, and you may need more sophisticated checking for a word
        # if any(c in string.ascii_letters for c in tokenizer.decode([next_token])) and tokenizer.decode(next_token).startswith(" "):
        # Do when the sentence ends.
        rand_f = random.random()
        if rand_f <= prob and any(c in ',.;!?' for c in tokenizer.decode([next_token])):
            emoji_token = np.random.choice(EMOJIS)
            emoji_encoded = tokenizer.encode(emoji_token, add_special_tokens=False)
            input_ids = torch.cat((input_ids, torch.Tensor([emoji_encoded])), dim=1)
            emoji_num += 1
        # Append the token to the input
        input_ids = torch.cat((input_ids, torch.Tensor([[next_token]])), dim=1)
        clean_ids = torch.cat((clean_ids, torch.Tensor([[next_token]])), dim=1)
        clean_text_ = tokenizer.decode(clean_ids[0].int())
        pattern = r"[^A-Za-z0-9.,;?!'\"()\-] "
        clean_text = re.sub(pattern, ' ', clean_text_).replace('\n', ' ')
        def replace_non_ascii(s):
            return ''.join([char if ord(char) < 128 else ' ' for char in s])
        clean_text = replace_non_ascii(clean_text)
        clean_text = re.sub(' +', ' ', clean_text)
        clean_text = re.sub(r'\.(?=[a-zA-Z])', '. ', clean_text)
        clean_text = re.sub(r'(?<=[a-zA-Z]) \.', '.', clean_text)
    dir_text = tokenizer.decode(input_ids[0].int())
    return clean_text, dir_text, emoji_num

subst_rule = SUBST_RULE
def generate_with_typo(args, prompt, tokenizer, model, prob):
    encoded_input = tokenizer(prompt, return_tensors='pt').to("cuda")
    input_ids = copy.deepcopy(encoded_input.input_ids).cpu()
    clean_ids = input_ids
    typo_num = 0
    MAX_ATMP = 10
    attemps = 0
    while len(input_ids[0]) < MAX_LENGTH and attemps < MAX_ATMP:
        with torch.no_grad():
            outputs = model(input_ids.cuda().int())
            next_token_logits = outputs.logits[:, -1, :].cpu()
            # Softmax and sample a token
            probs = nn.softmax(next_token_logits, dim=1)
            total = sum(probs[0].tolist())
            normalized_probs = [x/total for x in probs[0].tolist()]
            next_token = np.random.choice(len(probs[0]), p=normalized_probs)
            if next_token == tokenizer.eos_token_id:
                attemps += 1
                continue
        rand_f = random.random()
        next_token_str = tokenizer.decode([next_token])
        if rand_f <= prob and any(c in subst_rule.keys() for c in next_token_str):
            dirty_next_token_str = ""
            for char in next_token_str:
                if char in subst_rule.keys():
                    typo_num += 1
                    dirty_next_token_str += subst_rule[char]
                else:
                    dirty_next_token_str += char
            dirty_token = tokenizer.encode(dirty_next_token_str, add_special_tokens=False)
            input_ids = torch.cat((input_ids, torch.Tensor([dirty_token])), dim=1)
            clean_ids = torch.cat((clean_ids, torch.Tensor([[next_token]])), dim=1)
        else:
            input_ids = torch.cat((input_ids, torch.Tensor([[next_token]])), dim=1)
            clean_ids = torch.cat((clean_ids, torch.Tensor([[next_token]])), dim=1)
            
    clean_text_ = tokenizer.decode(clean_ids[0].int())
    pattern = r"[^A-Za-z0-9.,;?!'\"()\-] "
    clean_text = re.sub(pattern, ' ', clean_text_).replace('\n', ' ')
    def replace_non_ascii(s):
        return ''.join([char if ord(char) < 128 else ' ' for char in s])
    clean_text = replace_non_ascii(clean_text)
    clean_text = re.sub(' +', ' ', clean_text)
    clean_text = re.sub(r'\.(?=[a-zA-Z])', '. ', clean_text)
    clean_text = re.sub(r'(?<=[a-zA-Z]) \.', '.', clean_text)
    dir_text = tokenizer.decode(input_ids[0].int())
    # print("clean_text: ", clean_text, "\ndir_text: ", dir_text, "\ntypo_num: ", typo_num)
    return clean_text, dir_text, typo_num


def main(): 
    args = parse_args()
    
    out_path = os.path.join(args.input_path, args.domain) + "/" + args.gen_model_name+f"_test.{args.attack_method}{args.attack_args}_att.csv"
    print("*Output to file:", out_path)

    TESTSET_PATH = os.path.join(args.input_path, args.domain + "/" + args.gen_model_name + "_test.csv")
    df = pd.read_csv(TESTSET_PATH, sep="|")
    print("*Loaded from", TESTSET_PATH)

    matching_file = "multi_model_data/news_gptj_t1.5/id_matching.json"
    import json
    if os.path.exists(matching_file):
        print("*loading matches*")
        ordered_test_dataset_ori_hwt = []
        with open(matching_file, 'r') as file:
            matching_hwt_and_mgt = json.load(file)
    else:
        raise NotImplementedError

    model_name = args.model_list
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
    elif model_name in ["text-davinci-003","gpt-4"]:
        args.model = model_name
    else:
        args.model = model_name

    if model_name in ["text-davinci-003","gpt-4"]:
        print("generating samples with "+model_name)
    else:
        print("loading tokenizer "+ args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=CACHE_DIR)
        print("loading model " + args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=CACHE_DIR)
        if model_name == "llama13B":
            model = model.to(torch.bfloat16).cuda()
        else:
            model = model.cuda()


    try_times = MAX_TRIAL
    outputs = []
    rep_tot, gen_tot = 0,0
    attck_item_num = 0
    mgt_idx = 0
    from multi_model_data.repeating_detect import evaluate_text 
    hwts = df[df['label'] == 1]
    mgts = df[df['label'] == 0]
    for index, d in tqdm(df.iterrows()):
        if d["label"] == 1:
            continue
        seq = d["sequence"]
        try:
            for idx in range(try_times):
                prompt = get_prompt(seq) # Keep first 20 tokens to use as prompt.
                # print(model_name, args.attack_method)
                if model_name not in ["text-davinci-003","gpt-4"]:
                    if args.attack_method == "baseline":
                        decoded_output = baseline(args, prompt, tokenizer, model)
                    elif args.attack_method == "reprompt":
                        raise NotImplementedError
                        # decoded_output = reprompt(args, prompt, tokenizer, model)
                    elif args.attack_method == "emoji":
                        decoded_output, dirty_output, emoji_num = generate_with_emojis(args, prompt, tokenizer, model, float(args.attack_args))
                        attck_item_num += emoji_num
                        if index % 20 == 0:
                            print(prompt, '\n===\n', dirty_output, '\n===\n',decoded_output)
                            print("Emoji Number: ", attck_item_num/(gen_tot+1))
                    elif args.attack_method == "typo-cogen":
                        decoded_output, dirty_output, typo_num = generate_with_typo(args, prompt, tokenizer, model, float(args.attack_args))
                        attck_item_num += typo_num
                        if index % 20 == 0:
                            print(prompt, '\n===\n', dirty_output, '\n===\n',decoded_output)
                            print("Typo Number: ", attck_item_num/(gen_tot+1))
                    else:
                        raise NotImplementedError
                    decoded_output = truncate(decoded_output, 110)
                    token_num = count_tokens(decoded_output)
                    if token_num in range(100, 121):
                        df.at[index, "sequence"] = decoded_output
                        break
                    if idx == try_times-1:
                        print(token_num)
                        df.at[index, "sequence"] = "<blank text>"
                else:
                    if model_name == "gpt-4":
                        if args.attack_method == "icl":
                            pos_output = hwts.iloc[matching_hwt_and_mgt[mgt_idx]["hwt_id"]]["sequence"]
                            icl_prompt = f"""Instruction language: English\nInput language: English\nOutput language: English\nCategories: News Generation\nDefinition: In this task, based on the given input, we ask you to continue writing the news text for about 90 words. Don't repeat the exmaple outputs.\nPositive Examples:\nInput: {prompt}\nOutput: {pos_output}\nNegative Examples:\nInput: {prompt}\nOutput: {d["sequence"]}\n\nInput: {prompt}\nOutput: {prompt}"""
                            final_prompt = icl_prompt.strip()
                        elif args.attack_method == 'csgen':
                            subst_rule = ("a", "z")
                            final_prompt = f"Please continue this text in about 90 words, replace all the '{subst_rule[0]}'s into '{subst_rule[1]}'s and '{subst_rule[1]}'s into '{subst_rule[0]}'s:" +prompt.strip()
                        else:
                            final_prompt = "Please continue this text in about 90 words:" + prompt.strip()
                        dirty_ans = gpt4_completion(final_prompt)               
                        def subst_back(text):
                            cleaned_t = ""
                            for char in text:
                                if char == subst_rule[0]:
                                    cleaned_t += subst_rule[1]
                                elif char == subst_rule[1]:
                                    cleaned_t += subst_rule[0]
                                else:
                                    cleaned_t += char
                            return cleaned_t
                        ans = subst_back(dirty_ans)
                    else:
                        ans = davinci_completion(model_name, prompt)

                    if args.model == "gpt-4":
                        decoded_output = exp_truncate(prompt.strip() + " " + ans, 110)
                    else:
                        decoded_output = truncate(prompt.strip() + " " + ans, 110)
                    token_num = count_tokens(decoded_output)
                    if token_num in range(100, 121):
                        df.at[index, "sequence"] = decoded_output
                        break
                    if idx == try_times-1:
                        print(token_num)
                        df.at[index, "sequence"] = "<blank text>"
            mgt_idx += 1
        except RuntimeError:
            print("Runtime Error!")              
        gen_tot += 1
        rep = evaluate_text(decoded_output)
        if rep:
            print(f"***Repeating {rep} times***")
            rep_tot += rep
            print(f"===Repeating tot {rep_tot/gen_tot}={rep_tot}/{gen_tot} times===")
    print("Attack Number: ", attck_item_num / gen_tot) 
    df.to_csv(out_path, sep = "|", index = None)
    print("Writing csv file to " + out_path)
    torch.cuda.empty_cache()
 
if __name__ == "__main__":
    main()
