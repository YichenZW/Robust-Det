import os
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
)
from tqdm import tqdm
import torch
import csv

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


def main():
    args = parse_args()

    out_path = (
        os.path.join(args.input_path, args.domain)
        + "/"
        + args.gen_model_name
        + f"watermark_test.pegasus_para_prompt_att.csv"
    )

    print(out_path)

    TESTSET_PATH = os.path.join(
        args.input_path, args.domain + "/" + args.gen_model_name + f"_test.pegasus_para_prompt_att.csv"
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
        args.model = model_name
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
            if d["label"] == 1:
                output = {"sequence": d["sequence"], "label": d["label"]}
                csvwriter.writerows([output])
                outputs.append(output)
                continue
            seq = d["sequence"]
            try:
                if model_name not in ["text-davinci-003", "gpt-4"]:
                    # Keep only first 20 tokens to use as prompt.
                    prompt = d["prompt"]
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
                        # note that if the model is on cuda, then the input is on cuda
                        # and thus the watermarking rng is cuda-based.
                        # This is a different generator than the cpu-based rng in pytorch!
                        output_tokens = model.generate(
                            **tokenized_input,
                            logits_processor=LogitsProcessorList([watermark_processor]),
                            top_p=args.top_p,
                            top_k=0,
                            temperature=args.temp,
                            repetition_penalty=args.rp,
                            num_beams=args.num_beams,
                            do_sample=args.do_sample,
                            min_length=120,
                            max_length=500,
                            pad_token_id=tokenizer.eos_token_id # for gpt2 and gptj
                        )
                        # if decoder only model, then we need to isolate the
                        # newly generated tokens as only those are watermarked, the input/prompt is not
                        output_text = output_tokens[
                            :, tokenized_input["input_ids"].shape[-1] :
                        ]
                        decoded_output = (
                            prompt
                            + tokenizer.batch_decode(
                                output_text,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )[0]
                        )
                        decoded_output = truncate(decoded_output, 110)
                        token_num = count_tokens(decoded_output)
                        print(f"*Index {index} Attempt {idx}, #token {token_num}")
                        if token_num in range(100, 121):
                            df.at[index, "sequence"] = decoded_output
                            output = {"sequence": decoded_output, "label": d["label"]}
                            csvwriter.writerows([output])
                            outputs.append(output)
                            break
                        if idx == try_times - 1:
                            print(token_num)
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