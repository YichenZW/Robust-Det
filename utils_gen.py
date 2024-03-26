import nltk
import os
import pandas as pd
import jsonlines

all_lms = ["gpt1","gpt2sm","gpt2md","gpt2lg","gpt2xl","gpt3","gpt4","gptneosm","gptneomd","gptneolg","gptj","llama7B","llama13B"]

classifier_dict = {"albert":"albert-large-v2",
                    "electra":"google/electra-large-discriminator",
                    "roberta":"roberta-large",
                    "bert":"bert-large-cased"}

ckpt_dir_dict = {"news":"checkpoint",
           "review":"domains/review/classifier_ckpt",
           "wiki":"domains/wiki/classifier_ckpt"}

data_dir_dict = {"news":"data/electra_input/",
                 "review":"domains/review/classifier_input/",
                 "wiki":"domains/wiki/classifier_input/"}

home = "/home/"

predict_dir_dict = {"news":home + "data/classifier_predict",
                    "review":home + "domains/review/classifier_predict",
                    "wiki":home + "domains/wiki/classifier_predict"}

def read_txt(filepath):
    lines = []
    with open(filepath,"r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def count_tokens(text):
    toks = nltk.word_tokenize(text)
    return len(toks)

def truncate(text,tgt):
    """Delete the incomplete sentenence at the end."""
    last,now = 0,0
    text = text.strip()
    sens = nltk.sent_tokenize(text)
    sens_num = len(sens)
    if text[-1] not in ['.','"','!','?',';']:
        sens_num -= 1
    i = 0
    while i < sens_num:
        if now < tgt:
            last = now
            now += count_tokens(sens[i])
            i += 1
        else:
            break
    if tgt-last < now-tgt:
        out_sens = i-1
    else:
        out_sens = i
    res = " ".join(sens[:out_sens])
    return res


def get_prompt(text,prompt_len=20):
    toks = nltk.word_tokenize(text)
    return " ".join(toks[:prompt_len])

def createIfNotExist(path):
    path = os.path.expandvars(path)
    if not os.path.exists(path):
        os.mkdir(path)

def get_model_name(model_name,domain):
    if domain == "review":
        ckt_dir = "checkpoint/domains/review/"
    elif domain == "wiki":
        ckt_dir = "checkpoint/domains/wiki/"
    elif domain == "news":
        ckt_dir = "checkpoint/"
    
    if model_name in ["openai-gpt","gpt2","gpt2-medium","gpt2-large","gpt2-xl"]:
        return ckt_dir + model_name
    elif model_name == "gptneo-sm":
        return "EleutherAI/gpt-neo-125m"
    elif model_name == "gptneo-md":
        return "EleutherAI/gpt-neo-1.3B"
    elif model_name == "gptneo-lg":
        return "EleutherAI/gpt-neo-2.7B"
    elif model_name == "gptj":
        return "EleutherAI/gpt-j-6b"
    elif model_name == "llama7B":
        return "path/to/llama/converted/7B"
    elif model_name == "llama13B":
        return "path/to/llama/converted/13B"
    else:
        return model_name



def get_sorted_list(d, reverse=True):
    res = []
    for item in d.items():
        for AS in item[1]:
            res.append((item[0],AS))
    return sorted(res, key=lambda x:x[1], reverse=reverse)

def createIfNotExist(path):
    path = os.path.expandvars(path)
    if not os.path.exists(path):
        os.mkdir(path)

def write_jsonl(data,path):
    with jsonlines.open(path,"w") as writer:
        writer.write_all(data)

def get_vote(row,len):
    avg = float((row.sum()-row["ref"])/len)
    if avg < 0.5:
        return 0
    else:
        return 1

def clean(text):
    text = text.replace("<br />","")
    text = text.replace("\n","")
    text = text.replace("()","")
    text = text.replace("( )","")
    text = text.replace("(  )","")
    text = text.replace("(;)","")
    text = text.replace("(; )","")
    text = text.replace("(,)","")
    text = text.replace("(, )","")
    return text

