import numpy as np
import os
import random
import torch
import csv
import argparse
import transformers
import re
from attacks.perturb import perturb_texts
from attacks.paraphrase import paraphrase
from paraphrase_openai import paraphrase_openai_
from utils import load_csv
from tqdm import tqdm

def args_init():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.getcwd(), "multi_model_data"),
        type=str,
        help="",
    )
    parser.add_argument(
        "--dataset_name", 
        default="news_gptj_t1.5", 
        type=str)
    parser.add_argument(
        "--generate_model_name", 
        type=str, 
        default="gptj"
        )
    parser.add_argument('--pct_words_masked', 
        type=float, 
        default=0.6, 
        help="pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))")
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_perturbation_list', type=str, default="1,10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--base_model_name', type=str, default="EleutherAI/gpt-j-6b")
    parser.add_argument('--scoring_model_name', type=str, default="")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large", help="for perturbation.")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', 
                        type=float, default=0.96, help="Used both for ptb and pegasus, set as 0.96 for ptb.")
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--baselines_only', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0, help="for perturbation.")
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="/cache")
    parser.add_argument('--seed', type=int, default=567)
    parser.add_argument('--gpu_id', 
                        type=str, default="0")
    parser.add_argument('--device', 
                        type=str, default="cuda")
    parser.add_argument('--attack_method', 
                        type=str, 
                        default="typo_mix", 
                        help="ptb, pegasus, dipper, typo_(mix/trans/subst/delet/insert), homo_(ECES/ICES), form_(shift/zero-sp), chatgpt_para, word_subst_(modelfree/modelbase)")
    # for dipper
    parser.add_argument('--lex_diversity', 
                        type=int, default=60, help="0-100")   
    parser.add_argument('--order_diversity', 
                        type=int, default=60, help="0-100")   
    # for chatgpt3.5 paraphrasing
    parser.add_argument('--bigram',
                        type=bool, default=True)
    
    parser.add_argument('--watermark',
                        type=bool, default=False)
    args = parser.parse_args()
    return args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def typo_attack(args, texts, mode):
    # mode include replace, repeat, delete, trans
    LETTER_F = {"a":11.7, "b":4.4, "c":5.2, "d":3.2, "e":2.8, "f":4, "g":1.6, "h":4.2, "g":1.6, "h":4.2, "i":7.3, "j":0.51, "k":0.86, "l":2.4, "m":3.8, "n":2.3, "o":7.6, "p":4.3, "q":0.22, "r":2.8, "s":6.7, "t":16, "u":1.2, "v":0.82, "w":5.5, "x":0.045, "y":0.76, "z":0.045}
    # Mička, Pavel. "Letter frequency (English)". Algoritmy.net. Archived from the original on 4 March 2021. Retrieved 14 June 2022. Source is Leland, Robert. Cryptological mathematics. [s.l.] : The Mathematical Association of America, 2000. 199 p. ISBN 0-88385-719-7
    MIX_PROB = {"trans":0.011, "delet":0.23, "subst":0.556, "insert":0.203}
    att_texts = []
    for text in tqdm(texts):
        words = text.split()
        att_word_num = args.pct_words_masked * len(words)
        if att_word_num == 0:
            att_word_num = 1
        att_index = random.sample(list(range(len(words))), int(att_word_num)) 

        def trans(victim):
            w_id = random.choice(list(range(len(victim)-1)))
            victim[w_id], victim[w_id+1] = victim[w_id+1], victim[w_id]
        def subst(victim):
            w_id = random.choice(list(range(len(victim))))
            if victim[w_id].islower():
                victim[w_id] = random.choice('abcdefghijklmnopqrstuvwxyz')
            elif victim[w_id].isupper():
                victim[w_id] = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            #todo: skip if w_id is punctuation mark or number
        def delet(victim):
            w_id = random.choice(list(range(len(victim))))
            del victim[w_id]
        def insert(victim):
            w_id = random.choice(list(range(len(victim))))
            victim.insert(w_id, random.choice('abcdefghijklmnopqrstuvwxyz'))
        FUNC_DICT = {"trans":trans, "delet":delet, "subst":subst, "insert":insert}
        for att_id in att_index:
            victim = list(words[att_id])
            if len(victim) <= 1:
                continue
            if mode=="trans":
                trans(victim)
                victim = ''.join(victim)
                words[att_id] = victim
            elif mode=="subst":
                subst(victim)
                victim = ''.join(victim)
                words[att_id] = victim
            elif mode=="delet":
                delet(victim)
                victim = ''.join(victim)
                words[att_id] = victim
            elif mode=="insert":
                insert(victim)
                victim = ''.join(victim)
                words[att_id] = victim
            elif mode=="mix":
                keys = list(MIX_PROB.keys())
                values = list(MIX_PROB.values())
                sel_mode = random.choices(keys, weights=values, k=1)[0]
                sel_func = FUNC_DICT[sel_mode]
                sel_func(victim)
                victim = ''.join(victim)
                words[att_id] = victim
            else:
                raise NotImplementedError
        att_texts.append(" ".join(words))
    assert len(att_texts)==len(texts), "Length not match."
    return att_texts

def perturb_attack(args, texts, do_chatgpt=False):
    int8_kwargs = {}
    half_kwargs = {}
    if args.int8:
        int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
    elif args.half:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    print(f'Loading mask filling model {args.mask_filling_model_name}...')
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir=args.cache_dir).to(args.device)
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    # preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=args.cache_dir)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(args.mask_filling_model_name, model_max_length=n_positions, cache_dir=args.cache_dir)
    # if args.dataset in ['english', 'german']:
    #     preproc_tokenizer = mask_tokenizer
    if do_chatgpt==True:
        mask_model="gpt-3.5-turbo"
        mask_tokenizer=None
    outputs = perturb_texts(args, texts, mask_model, mask_tokenizer, args.span_length, args.pct_words_masked, ceil_pct=False)
    return outputs

def homo_attack(args, texts, mode): 
    # mode can be ICES or ECES
    if mode == "ECES":
        from VIPER.viper_eces import eces 
        outputs = eces(args.pct_words_masked, texts) # pct_words_masked > 0 at word level
        return outputs
    elif mode == "ICES":
        from VIPER.viper_ices import ices
        outputs = ices(args.pct_words_masked, texts)
        return outputs
    else:
        raise ValueError

from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

pronouns = ["I", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them", "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs", "this", "that", "these", "those", "who", "whom", "whose", "which", "what", "myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves"]
other_function_words = ["a", "an", "the", "and", "but", "or", "nor", "for", "yet", "so", "as", "if", "is", "are", "be", "was", "were", "being", "been"]
stop_words = pronouns + other_function_words
def synonym_subst(args, text, mode):
    words = re.findall(r"[\w']+|[.,!?;]", text)
    subst_num = round(args.pct_words_masked * len(words))
    print(subst_num)
    for i in range(subst_num):
        retry = 5
        while retry: 
            subject_word_idx = random.choice(list(range(len(words))))
            subject_word = words[subject_word_idx]
            if subject_word in ".,!?;":
                retry -= 1
                continue
            if subject_word in stop_words:
                retry -= 1
                continue
            synonym_list = get_synonyms(subject_word)
            if len(synonym_list) == 0:
                retry -= 1
                continue
            synonym = synonym_list[0]
            if synonym == words[subject_word_idx]:
                retry -= 1
                continue    
            synonym = synonym.replace("_", " ")
            # print(f"{words[subject_word_idx]} -> {synonym}")
            words[subject_word_idx] = synonym
            break
    substed_text = " ".join(words)
    return substed_text

def format_attack(args, texts, mode):
    # mode can be shift/shift-u/space/zero-sp/
    # if attack sentence number is 0, then only insert *one* attach character

    if mode == "shift-u":
        import nltk
        nltk.download('punkt') 
        from nltk.tokenize import sent_tokenize
        
        att_texts = []
        tot_att_num = 0
        for text in tqdm(texts):
            sentences = sent_tokenize(text)
            att_sentences_num = args.pct_words_masked * len(sentences)
            if args.pct_words_masked == 0:
                att_sentences_num = 1
            if args.pct_words_masked <= 1:
                att_index = random.sample(list(range(len(sentences))), round(att_sentences_num)) 
            else: 
                att_index = list(range(len(sentences))) + random.sample(list(range(len(sentences))), round(att_sentences_num)-len(sentences))
            tot_att_num += round(att_sentences_num)
         
            for att_id in att_index:
                sentences[att_id] = sentences[att_id] + " \u000B\u000B "
            att_text = "".join(sentences)
            att_texts.append(att_text)
        print(tot_att_num / len(texts) * 2)

    elif mode == "zero-sp":
        att_texts = []
        for text in tqdm(texts):
            words = text.split()
            att_word_num = args.pct_words_masked * len(words)
            if att_word_num == 0:
                att_word_num = 1
            att_index = random.sample(list(range(len(words))), int(att_word_num))
            for att_id in att_index:
                victim = list(words[att_id])
                w_id = random.choice(list(range(len(victim))))
                victim.insert(w_id, '\u200B') 
                victim = ''.join(victim)
                words[att_id] = victim
            att_texts.append(" ".join(words))    
    assert len(att_texts)==len(texts), "Length not match."
    return att_texts

def chatgpt_para(args, texts):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    new_texts, paras = paraphrase_openai_(texts, tokenizer, bigram=args.bigram)
    return new_texts, paras


def main():
    args = args_init()
    set_seed(args)
    # test CUDA (bug fixing for the MIT machine)
    try:
        args.gpu_id = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print("CUDA_VISIBLE_DEVICES: ", args.gpu_id, torch.cuda.get_device_name())
    except:
        args.gpu_id = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print("CUDA_VISIBLE_DEVICES: ", args.gpu_id, torch.cuda.get_device_name())

    # Load Dataset
    TESTSET_PATH = os.path.join(args.data_dir, args.dataset_name + "/" + args.generate_model_name + "_test.csv")
    test_dataset  = load_csv(TESTSET_PATH)
    # !IMPORTANT: We only attack MGTs!
    texts = [d['sequence'] for d in test_dataset if d['label']=='0']
    labels = [d['label'] for d in test_dataset if d['label']=='0']

    if args.attack_method == "ptb":
        ATTACKED_PATH = os.path.join(args.data_dir, args.dataset_name + "/" + args.generate_model_name + f"_test.{args.attack_method+str(args.pct_words_masked)}_att.csv")
    elif args.attack_method == "pegasus":
        ATTACKED_PATH = os.path.join(args.data_dir, args.dataset_name + "/" + args.generate_model_name + f"_test.{args.attack_method+str(args.pct_words_masked)}_att.csv")     
    elif args.attack_method == "dipper":
        ATTACKED_PATH = os.path.join(args.data_dir, args.dataset_name + "/" + args.generate_model_name + f"_test.{args.attack_method}_L{str(args.lex_diversity)}O{str(args.order_diversity)}_att.csv")  
    elif args.attack_method[:4] == "typo": 
        ATTACKED_PATH = os.path.join(args.data_dir, args.dataset_name + "/" + args.generate_model_name + f"_test.{args.attack_method+str(args.pct_words_masked)}_att.csv") 
    elif args.attack_method[:4] == "homo":               
        ATTACKED_PATH = os.path.join(args.data_dir, args.dataset_name + "/" + args.generate_model_name + f"_test.{args.attack_method+str(args.pct_words_masked)}_att.csv") 
    elif args.attack_method[:4] == "form": 
        ATTACKED_PATH = os.path.join(args.data_dir, args.dataset_name + "/" + args.generate_model_name + f"_test.{args.attack_method+str(args.pct_words_masked)}_att.csv")  
    elif args.attack_method[:12] == "chatgpt_para":
        ATTACKED_PATH = os.path.join(args.data_dir, args.dataset_name + "/" + args.generate_model_name + f"_test.{args.attack_method}_bi{args.bigram}_att.csv")   
    elif args.attack_method[:10] == "word_subst":
        ATTACKED_PATH = os.path.join(args.data_dir, args.dataset_name + "/" + args.generate_model_name + f"_test.{args.attack_method+str(args.pct_words_masked)}_att.csv")        
    print("Attack: ", ATTACKED_PATH)

    # Begin Attack    
    attacked_texts_split_sents = None
    if args.attack_method == "ptb":
        attacked_texts = perturb_attack(args, texts)
    elif args.attack_method in ["dipper", "pegasus"]:
        attacked_texts, attacked_texts_split_sents = paraphrase(args, texts)
    elif args.attack_method[:4] == "typo": 
        attacked_texts = typo_attack(args, texts, args.attack_method.split("_")[1])
    elif args.attack_method[:4] == "homo": 
        attacked_texts = []
        for text in tqdm(texts):
            attacked_texts.append(homo_attack(args, text, args.attack_method.split("_")[1]))
    elif args.attack_method[:4] == "form":
        attacked_texts = format_attack(args, texts, args.attack_method.split("_")[1])
    elif args.attack_method[:12] == "chatgpt_para":
        _, attacked_texts = chatgpt_para(args, texts) 
    elif args.attack_method[:10] == "word_subst":
        if args.attack_method[:20] == "word_subst_modelfree":
            attacked_texts = []
            for text in tqdm(texts):
                attacked_texts.append(synonym_subst(args, text, args.attack_method.split("_")[1]))
        elif args.attack_method[:20] == "word_subst_modelbase":
            from attacks.word_subst_modelbase import generate_attack_with_lm_replacement
            
            sub_parser = argparse.ArgumentParser()
            sub_parser.add_argument('--test_ratio', 
                            type=float, default=args.pct_words_masked)
            sub_parser.add_argument('--num_replacement_retry', 
                            type=int, default=3)
            sub_parser.add_argument('--attack_method', 
                            type=str, default='')
            # No use
            sub_parser.add_argument('--dataset_name', 
                            type=str, default='')
            sub_parser.add_argument('--watermark', 
                            type=str, default='')
            sub_parser.add_argument('--pct_words_masked', 
                            type=str, default='')
            sub_parser.add_argument('--generate_model_name', 
                            type=str, default='')            
            sub_args = sub_parser.parse_args()
            sub_args.attack_method = 'llama_replacement'
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print("*Loading Llama*")
            replacement_model = AutoModelForCausalLM.from_pretrained('/home/gridsan/tianxing/txml_shared/llama/llama/7B_hf').to("cuda")
            replacement_tokenizer = AutoTokenizer.from_pretrained('/home/gridsan/tianxing/txml_shared/llama/llama/7B_hf')

            attacked_texts, subword_num = generate_attack_with_lm_replacement(texts, sub_args, replacement_model, replacement_tokenizer)
            print("***Average Substituted Word Number:", sum(subword_num)/len(subword_num))
        else:
            raise NotImplementedError

    if attacked_texts_split_sents is None:
        attacked_dataset = [{"sequence": att, "label": l} for att, l in zip(attacked_texts, labels)]
        # Save into csv dataset
        with open(ATTACKED_PATH, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = attacked_dataset[0].keys(), delimiter="|")
            # writing headers (field names)
            writer.writeheader()    
            # writing data rows
            writer.writerows(attacked_dataset)
        print("*Success saved to", ATTACKED_PATH)
    else: 
        import json
        attacked_dataset = [{"sequence": att, "label": l, "split_sents": attss} for att, l, attss in zip(attacked_texts, labels, attacked_texts_split_sents)]
        ATTACKED_PATH_JSON = os.path.join(args.data_dir, args.dataset_name + f"/{args.generate_model_name}_test.pegasus_att.cache.json")
        with open(ATTACKED_PATH_JSON, 'w') as file:
            json.dump(attacked_dataset, file)
        print("*Success saved to", ATTACKED_PATH_JSON)    

if __name__ == "__main__":
    main()