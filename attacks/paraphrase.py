import time
import os
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm, trange
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
)
import json
import random
random.seed(0)
CACHE_DIR = "/cache"
def paraphrase(args, texts):
    if args.attack_method == "pegasus":
        pegasus_cache_dir = os.path.join(args.data_dir, args.dataset_name + f"/{args.generate_model_name}_test.pegasus_att.cache.json")
        if os.path.exists(pegasus_cache_dir):
            print("*loading pegasus cache*")
            with open(pegasus_cache_dir, 'r') as file:
                pegasus_cache = json.load(file)
            output = []
            for data in tqdm(pegasus_cache):
                para_sent_num = 0
                join_sequence = ""
                for sent_org, sent_para in zip(data['split_sents'][0], data['split_sents'][1]):
                    if random.random() <= args.pct_words_masked:
                        join_sequence += sent_para + " "
                        para_sent_num += 1
                    else: 
                        join_sequence += sent_org + " "
                
                output.append(join_sequence)
                
            return output, None
        else:
            print("*failed loading pegasus cache, generating*")
            output, output_spilt_sents = paraphrase_gen(args, texts, top_p=args.top_p)
            return output, output_spilt_sents
    elif args.attack_method == "dipper":
        print("Loading DIPPER Model...")
        dp = DipperParaphraser()
        output = []
        for input_text in tqdm(texts, desc="Dippering"):
            output_sample = dp.paraphrase(input_text, lex_diversity=args.lex_diversity, order_diversity=args.order_diversity, prefix="", do_sample=True, top_p=0.96, top_k=None, max_length=512)
            output.append(output_sample)
    else:
        raise NotImplementedError

    return output, None

def paraphrase_gen(args, dataset, paraphraser_name="tuner007/pegasus_paraphrase", sample=True, top_p=0.96):
    # if use local_files_only = True, it will load the cache instead, which will not work because the cache directory has invalid characters

    device = args.device
    max_length = 60
    paraphraser = PegasusForConditionalGeneration.from_pretrained(
        paraphraser_name, cache_dir=CACHE_DIR).to(device)
    paraphraser_tokenizer = PegasusTokenizer.from_pretrained(paraphraser_name, cache_dir=CACHE_DIR)

    def paraphrase(sents):
        '''
        Arguments:
            sents: list of sentences (max len under max length!)
        Returns:
            paraphrased: list of paraphrased sents
        '''
        # Baceapp Corp. and LightSpin Technologies will move to BU's Engineering and Science building."
        batch = paraphraser_tokenizer(
            sents, truncation=True, padding='longest', return_tensors="pt", max_length=max_length).to(device)
        if sample:
            paraphrased_ids = paraphraser.generate(
                **batch, max_length=max_length, temperature=1, top_p=top_p, no_repeat_ngram_size=3, do_sample=True)
        else:
            paraphrased_ids = paraphraser.generate(
                **batch, max_length=max_length, temperature=1.5, num_beams=10, no_repeat_ngram_size=3)
        paraphrased = paraphraser_tokenizer.batch_decode(
            paraphrased_ids, skip_special_tokens=True)

        return paraphrased

    sents, data_len = [], []
    for text in tqdm(dataset, desc="Tokenizer"):
        sent_list = sent_tokenize(text)
        sents.extend(sent_list)
        data_len.append(len(sent_list))

    batch_size = 64

    start_pos = 0
    paraphrased_sents = []
    for batch_id in trange((sum(data_len)-1)//batch_size +1, desc="Paraphrase"):
            
        temp_sents = sents[start_pos: min(start_pos+batch_size, len(sents))]
        paraphrased_sents.extend(paraphrase(temp_sents))
        start_pos += batch_size
    
    start_pos = 0
    output = []
    output_split_sents = []
    for l in data_len:
        output.append(' '.join(paraphrased_sents[start_pos: start_pos+l]))
        output_split_sents.append((sents[start_pos: start_pos+l],paraphrased_sents[start_pos: start_pos+l]))
        start_pos += l
    
    return output, output_split_sents

class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model)

        self.model = self.model.to(torch.bfloat16).cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            
            final_input_text = f'lexical = {lex_code}, order = {order_code}'
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f' <sent> {curr_sent_window} </sent>'

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            
            output_text += " " + outputs[0]

        return output_text

    
