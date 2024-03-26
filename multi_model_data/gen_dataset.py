import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
import argparse
from utils_gen import get_prompt, count_tokens, truncate
import os
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
import openai
from openai import OpenAI
import nltk
import random
CACHE_DIR = "your/path/to/cache"

SPLIT = "train"

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
        help="for loading dataset to get the human-written texts as reference, match with file name.",
    )
    parser.add_argument(
        "--gen_model_name",
        type=str,
        default="Llama2-7b-hf",
        help="name for model or saved dataset, i.e., gpt-3.5-turbo-instruct, Llama2-7b-hf, gptj, text-davinci-003, gpt2-xl, ..."
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
        help="official model name in Huggingface/other API. Will generate automatically.",
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
        default=1.0,
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

def gpt4_completion(prompt):
    args = parse_args()
    client = OpenAI(
    api_key="your/openai/api",
    )
    messages = [{"role": "user", "content": "Please continue this text in about 90 words:"+prompt.strip()}]
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4",
        top_p=args.top_p,
        temperature=args.temp,
        max_tokens=160
    )
    ans = response.choices[0].message.content
    return ans

def davinci_completion(model_name, prompt):
    args = parse_args()
    openai.api_key = "your/openai/api"
    if model_name=="davinci":
        model_name="text-davinci-003"
    response = openai.Completion.create(
                        model=model_name,
                        prompt="Please continue this text in about 90 words: "+prompt.strip(),
                        top_p=args.top_p,
                        temperature=args.temp,
                        max_tokens=500
                    )
    ans = response["choices"][0]["text"]
    return ans

def get_prompt(text, prompt_len=20):
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens[:prompt_len])

def main(): 
    args = parse_args()
    if args.gen_model_name not in ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-4"]:

        args.device = torch.device("cuda" if torch.cuda.is_available() else ValueError)
        print("Current Device:", args.device)
    
    out_path = os.path.join(args.input_path, args.domain +"_"+ args.gen_model_name +"_t"+ str(args.temp)) + "/" + args.gen_model_name + f"_{SPLIT}.csv"
    if not os.path.exists(os.path.join(args.input_path, args.domain +"_"+ args.gen_model_name +"_t"+ str(args.temp))):
        os.mkdir(os.path.join(args.input_path, args.domain +"_"+ args.gen_model_name +"_t"+ str(args.temp)))
    print("*Output to file:", out_path)
    TESTSET_PATH = os.path.join(args.input_path, args.domain + "/" + args.load_dataset_model + f"_{SPLIT}.csv")

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
    elif model_name == "Llama2-7b-hf":
        args.model = "meta-llama/Llama-2-7b-hf"
    elif model_name in ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-4"]:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        args.model = model_name

    if model_name in ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-4"]:
        print("generating samples with "+model_name)
    else:
        print("loading tokenizer "+ args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=CACHE_DIR)
        print("loading model " + args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=CACHE_DIR)
        if model_name == "llama13B" or model_name == "Llama2-7b-hf":
            model = model.to(torch.bfloat16).cuda()
            """
            def generate(
                self,
                prompt_tokens: List[List[int]],
                max_gen_len: int,
                temperature: float = 0.6,
                top_p: float = 0.9,
                logprobs: bool = False,
                echo: bool = False,
            ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
            """
        else:
            model = model.cuda()

    try_times = MAX_TRIAL
    outputs = []
    rep_tot, gen_tot = 0, 0
    for index, d in tqdm(df.iterrows()):
        if d["label"] == 1: # Human written texts
            continue
        seq = d["sequence"]
        prompt = get_prompt(seq)

        if model_name not in ["text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-4"]:
            
            encoded_input = tokenizer(prompt, return_tensors='pt').to("cuda")
            for idx in range(try_times):
                output = model.generate(encoded_input.input_ids,
                                        top_p=args.top_p,
                                        top_k=0,
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
                decoded_output = truncate(decoded_output,110)
                token_num = count_tokens(decoded_output)
                
                if token_num in range(100,121):
                    df.at[index, "sequence"] = decoded_output
                    break
                if idx == try_times-1:
                    print(token_num)
                    df.at[index, "sequence"] = "<blank text>"
                
        else:
            for idx in range(try_times):
                if model_name == "gpt-4":
                    ans = gpt4_completion(prompt)
                else:
                    ans = davinci_completion(model_name, prompt)

                if args.model == "gpt-4":
                    decoded_output = exp_truncate(prompt.strip() + " " + ans, 110)
                else:
                    decoded_output = truncate(prompt.strip() + " " + ans,110)
                token_num = count_tokens(decoded_output)
                print(f"*Attempt {idx}, #token {token_num}: {decoded_output}")
                if token_num in range(100, 121):
                    df.at[index, "sequence"] = decoded_output
                    break
                if idx == try_times-1:
                    print(token_num)
                    df.at[index, "sequence"] = "<blank text>"
        gen_tot += 1
        from repeating_detect import evaluate_text 
        rep = evaluate_text(decoded_output)
        if rep:
            print(f"***Repeating {rep} times***")
            rep_tot += rep
            print(f"===Repeating tot {rep_tot/gen_tot}={rep_tot}/{gen_tot} times===")
        
        df.to_csv(out_path, sep = "|", index = None)
    print("Writing csv file to " + out_path)

if __name__ == "__main__":
    main()