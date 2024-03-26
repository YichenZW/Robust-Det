import os
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
)
from tqdm import tqdm
import torch
import torch.nn.functional as nn
import numpy as np
import csv
import re
import sys
import argparse
from utils_gen import get_prompt, count_tokens, truncate, createIfNotExist
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
import openai
import nltk
import random
import copy
MAX_TRIAL = 10


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--domain",
        type=str,
        default="news_gptj_t1.5",
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
        default="emoji",
        help="emoji or typo",
    )
    parser.add_argument(
        "--attack_ratio",
        type=float,
        default=1.0,
        help="",
    )
    
    args = parser.parse_args()
    return args


def seed_everything(seed):
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    # np.random.seed(seed)          # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)


seed_everything(1)


def exp_truncate(text, tgt):
    now = 0
    sens = nltk.sent_tokenize(text)
    for i in range(len(sens)):
        if now < tgt:
            last = now
            now += count_tokens(sens[i])
        else:
            break
    if tgt - last < now - tgt:
        out_sens = i
    else:
        out_sens = i + 1
    res = " ".join(sens[:out_sens])
    return res

def get_prompt(text, prompt_len=20):
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens[:prompt_len])

from emoji_loader import emoji_list
EMOJIS = emoji_list().list

def main():
    args = parse_args()

    cuda_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = torch.cuda.memory_allocated(0)
    free_memory = cuda_memory - available_memory

    print(f"{free_memory} = {cuda_memory} - {available_memory}")


    out_path = (
        os.path.join(args.input_path, args.domain)
        + "/"
        + args.gen_model_name
        + f"_watermark_t{args.temp}_test.{args.attack}-cogen_{args.attack_ratio}.csv"
    )

    print(out_path)

    TESTSET_PATH = os.path.join(
        args.input_path, args.domain + "/" + args.gen_model_name + f"_test.csv"
    )

    df = pd.read_csv(TESTSET_PATH, sep="|")

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
    elif model_name in ["text-davinci-003", "gpt-4"]:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        args.model = model_name

    if model_name in ["text-davinci-003", "gpt-4"]:
        print("generating samples with " + model_name)
    else:
        print("loading tokenizer " + args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
        print("loading model " + args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir)
        if model_name == "llama13B":
            model = model.to(torch.bfloat16).cuda()
        else:
            model = model.cuda()

    try_times = 10
    outputs = []
    with open(out_path, 'a', newline='') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=["sequence", "label"], delimiter='|')

        csvwriter.writeheader()

        for index, d in tqdm(df.iterrows()):
            if index > 128:
                break
            if d["label"] == 1:
                output = {"sequence": d["sequence"], "label": d["label"]}
                csvwriter.writerows([output])
                outputs.append(output)
                continue
            seq = d["sequence"]
            try:
                if model_name not in ["text-davinci-003", "gpt-4"]:
                    # Keep only first 20 tokens to use as prompt.
                    prompt = get_prompt(seq)
                    print(f"prompt: {prompt}")
                    for idx in range(try_times):
                        # Watermarking
                        from extended_watermark_processor import WatermarkLogitsProcessor

                        watermark_processor = WatermarkLogitsProcessor(
                            vocab=list(tokenizer.get_vocab().values()),
                            gamma=0.25,
                            delta=2.0,
                            seeding_scheme="selfhash",
                        )  # equivalent to `ff-anchored_minhash_prf-4-True-15485863`
                        # Note:
                        # You can turn off self-hashing by setting the seeding scheme to `minhash`.

                        tokenized_input = tokenizer(prompt, return_tensors="pt").to(
                            model.device
                        )
                        input_ids = copy.deepcopy(tokenized_input.input_ids).cpu()
                        clean_ids = input_ids
                        typo_num, emoji_num = 0, 0
                        prob = args.attack_ratio
                        MAX_LENGTH = 160
                        attemps = 0

                        if args.attack=="typo":
                            subst_rule = {"c":"k", "k":"c"}

                            while len(input_ids[0]) < MAX_LENGTH and attemps < MAX_TRIAL:
                                with torch.no_grad():
                                    model_outputs = model.generate(input_ids.cuda().int(),
                                                    logits_processor=LogitsProcessorList([watermark_processor]),
                                                    top_p=args.top_p,
                                                    top_k=0,
                                                    temperature=args.temp,
                                                    repetition_penalty=args.rp,
                                                    num_beams=args.num_beams,
                                                    do_sample=args.do_sample,
                                                    max_new_tokens=3,
                                                    pad_token_id=tokenizer.eos_token_id)
                                    next_token = model_outputs[:, input_ids.shape[-1] : input_ids.shape[-1] + 1]
                                    if next_token == tokenizer.eos_token_id:
                                        attemps += 1
                                        continue
                                rand_f = random.random()
                                next_token_str = tokenizer.batch_decode(
                                                next_token,
                                                skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False,
                                                )[0]
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
                            if typo_num > 0:
                                print("\n===\n**clean_text: ", clean_text, "\n**dir_text: ", dir_text, "\n**typo_num: ", typo_num, "\n===\n")
                        elif args.attack == "emoji":
                            emoji_num = 0
                            while len(input_ids[0]) < MAX_LENGTH:
                                with torch.no_grad():
                                    model_outputs = model.generate(input_ids.cuda().int(),
                                                    logits_processor=LogitsProcessorList([watermark_processor]),
                                                    top_p=args.top_p,
                                                    top_k=0,
                                                    temperature=args.temp,
                                                    repetition_penalty=args.rp,
                                                    num_beams=args.num_beams,
                                                    do_sample=args.do_sample,
                                                    max_new_tokens=3,
                                                    pad_token_id=tokenizer.eos_token_id)
                                    next_token = model_outputs[:, input_ids.shape[-1] : input_ids.shape[-1] + 1]
                                    if next_token == tokenizer.eos_token_id:
                                        attemps += 1
                                        continue
                                # If the generated token is a word (you'll need a way to check this), append an emoji
                                # This is a simplified check, and you may need more sophisticated checking for a word
                                # if any(c in string.ascii_letters for c in tokenizer.decode([next_token])) and tokenizer.decode(next_token).startswith(" "):
                                # Do when the sentence ends.
                                rand_f = random.random()
                                next_token_str = tokenizer.batch_decode(
                                                next_token,
                                                skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False,
                                                )[0]           
                                if rand_f <= prob and any(c in ',.;!?' for c in next_token_str):
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
                            if emoji_num > 0:
                                print("\n===\n**clean_text: ", clean_text, "\n**dir_text: ", dir_text, "\n**typo_num: ", emoji_num, "\n===\n")
                        else:
                            raise NotImplementedError
                        # note that if the model is on cuda, then the input is on cuda
                        # and thus the watermarking rng is cuda-based.
                        # This is a different generator than the cpu-based rng in pytorch!
                        decoded_output = clean_text
                        decoded_output = truncate(decoded_output, 110)
                        token_num = count_tokens(decoded_output)
                        print(f"*=Attempt {idx}, #token {token_num}")
                        if token_num in range(100, 121):
                            df.at[index, "sequence"] = decoded_output
                            output = {"sequence": decoded_output, "label": d["label"]}
                            csvwriter.writerows([output])
                            outputs.append(output)
                            break
                        if idx == try_times - 1:
                            decoded_output = "<blank text>"
                            df.at[index, "sequence"] = decoded_output
                            output = {"sequence": decoded_output, "label": d["label"]}
                            csvwriter.writerows([output])
                            outputs.append(output)

                else:
                    raise NotImplementedError
                    for idx in range(try_times):
                        if model_name == "gpt-4":
                            ans = gpt4_completion(prompt)
                        else:
                            ans = davinci_completion(prompt)

                        if args.model == "gpt-4":
                            decoded_output = exp_truncate(prompt.strip() + " " + ans, 110)
                        else:
                            decoded_output = truncate(prompt.strip() + " " + ans, 110)
                        token_num = count_tokens(decoded_output)

                        if token_num in range(100, 120):
                            outputs.append(decoded_output)
                            break
                        if idx == try_times - 1:
                            outputs.append("")
            except RuntimeError:
                raise RuntimeError
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
