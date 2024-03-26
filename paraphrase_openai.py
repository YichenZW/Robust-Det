import openai
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from datasets import load_from_disk, Dataset
import argparse
from transformers import AutoTokenizer
from paraphrase_gen_util import accept_by_bigram_overlap, extract_list
import time

openai.api_key = "your/key"

def gen_prompt(sent, context):
  prompt = f'''Previous context: {context} \n Current sentence to paraphrase: {sent}'''
  return prompt

def gen_bigram_prompt(sent, context):
  prompt = f'''Previous context: {context} \n Paraphrase in 20 different ways and return a numbered list : {sent}'''
  return prompt
  
def query_openai(prompt):
  while True:
    try:
      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
          {
            "role": "user",
            "content": prompt
          }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
      return response.choices[0].message.content
    except openai.error.RateLimitError:
      time.sleep(5)
    except openai.error.APIError:
      time.sleep(2)
  
def query_openai_bigram(prompt):
  while True:
    try:
      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
          {
            "role": "user",
            "content": prompt
          }
        ],
        temperature=1,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
      return response.choices[0].message.content
    except openai.error.RateLimitError:
      time.sleep(5)
    except openai.error.APIError:
      time.sleep(2)


def paraphrase_openai_(texts, tokenizer, bigram=False):
  new_texts = []
  paras = []
  for text in tqdm(texts, desc="Tokenizer"):
    sents = sent_tokenize(text)
    para = []
    for i in range(len(sents)):
      sent = sents[i]
      context = sents[:i]
      if bigram:
        prompt = gen_bigram_prompt(sent, context)
        para_str = query_openai_bigram(prompt)
        para_ls = extract_list(para_str)
        if len(para_ls) < 20:
          print(para_str)
          print(para_ls)
          continue
        para.append(accept_by_bigram_overlap(sent, para_ls, tokenizer)) # 
      else:
        prompt = gen_prompt(sent, context) 
        para_sen = query_openai(prompt)
        para.append(para_sen) 
    new_texts.append(sents)
    paras.append(" ".join(para))
  return new_texts, paras
