import torch
from tqdm import tqdm 
from string2string.distance import LevenshteinEditDistance
edit_dist = LevenshteinEditDistance()
from string2string.misc import Tokenizer
tokenizer = Tokenizer(word_delimiter=' ')

from string2string.misc import ModelEmbeddings
from string2string.similarity import CosineSimilarity
cosine_similarity = CosineSimilarity()
from string2string.similarity import JaroSimilarity
jaro_similarity = JaroSimilarity()
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast
CACHE_DIR = "/cache"

def compute_average_std(input_list):
    if len(input_list) == 0:
        return None, None
    avg = statistics.mean(input_list)
    std_dev = statistics.stdev(input_list)
    return avg, std_dev

def dist_and_sim(dataset_org_mgt, dataset_att_mgt, dataset_ori_hwt, do_model_based=True):

    print("*** Computing Distance and Similarity ...***")
    edit_dists, jaro_sim, cos_sim, gpt2l_ppl, llama_ppl, mauves, bart_ss = [], [], [], [], [], [], []
    text1s, text2s, texths = [], [], []
    if do_model_based:
        bart_model = ModelEmbeddings(
            model_name_or_path='facebook/bart-large',
            device="cuda"
        )
    for d1, d2, dh in tqdm(zip(dataset_org_mgt, dataset_att_mgt, dataset_ori_hwt)):
        assert d1["label"] == d2["label"], "Wrong Pairing"
        assert d1["label"] != dh["label"], "Wrong Pairing"

        text1 = d1["sequence"]
        text1s.append(text1)
        text2 = d2["sequence"]
        if text2 == '':
            text2 = '<blank text>'
        text2s.append(text2)
        texth = dh["sequence"]
        texths.append(texth)

        text1_tokens = tokenizer.tokenize(text1)
        text2_tokens = tokenizer.tokenize(text2)

        edit_dist_score  = edit_dist.compute(text1_tokens, text2_tokens)

        edit_dists.append(edit_dist_score)

        # Cos Sim based on BART. Compute the sentence embeddings for each sentence
        if do_model_based:
            embeds = []
            for sentence in [text1, text2]:
                embedding = bart_model.get_embeddings(sentence, embedding_type='mean_pooling')
                embeds.append(embedding)
            result = cosine_similarity.compute(embeds[0], embeds[1], dim=1).item()
            cos_sim.append(result) 

        # Compute the Jaro similarity scores of the two versions of Hamlet at the word level
        jaro_similarity_hamlet_word_level = jaro_similarity.compute(text1_tokens, text2_tokens)
        jaro_sim.append(jaro_similarity_hamlet_word_level)

    torch.cuda.empty_cache()
    if do_model_based: 
        # PPL on Llama 7B
        print("*** Computing Llama-2 PPL ...***")
        print("*** Loading ***")
        llama_tokenizer = AutoTokenizer.from_pretrained('/path/to/llama/llama/7B_hf')
        llama_model = AutoModelForCausalLM.from_pretrained('/path/to/llama/llama/7B_hf', torch_dtype=torch.float16).to("cuda")
        for text2 in tqdm(text2s):
            with torch.no_grad():
                tokenized = llama_tokenizer(text2, return_tensors="pt").to("cuda")
                labels = tokenized.input_ids
                llama_ppl.append(-llama_model(**tokenized, labels=labels).loss.item())
        
        del llama_tokenizer, llama_model
        torch.cuda.empty_cache()

        # PPL on GPT-2 Large
        print("*** Computing GPT-2L PPL ...***")
        print("*** Loading ***")
        gpt2l_model = AutoModelForCausalLM.from_pretrained("gpt2-large", cache_dir=CACHE_DIR).to("cuda")
        gpt2l_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large", cache_dir=CACHE_DIR)
        for text2 in tqdm(text2s):
            with torch.no_grad():
                tokenized = gpt2l_tokenizer(text2, return_tensors="pt").to("cuda")
                labels = tokenized.input_ids
                gpt2l_ppl.append(-gpt2l_model(**tokenized, labels=labels).loss.item())
        
        del gpt2l_tokenizer, gpt2l_model
        torch.cuda.empty_cache()

        # BART Score
        print("*** Computing BARTScore ...")
        from bart_score import BARTScorer
        bart_scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')

        bart_ss_2h = bart_scorer.score(text2s, texths, batch_size=16)
        bart_ss_ab = bart_scorer.score(text2s, text1s, batch_size=16)

        del BARTScorer

        # BERT Score
        import bert_score
        bert_ss_2h = bert_score.score(
                text2s, texths, 
                rescale_with_baseline=True, lang="en"
            )[0].tolist()
        bert_ss_ab = bert_score.score(
                text2s, text1s, 
                rescale_with_baseline=True, lang="en"
            )[0].tolist()
        
        do_mauve = True
        if do_mauve:
            print("*** Computing MAUVE ...")
            import mauve
            mauve_ab = float(mauve.compute_mauve(p_text=text1s, q_text=text2s, device_id=0, max_text_length=1023, verbose=False).mauve)
            mauve_2h = float(mauve.compute_mauve(p_text=text2s, q_text=texths, device_id=0, max_text_length=1023, verbose=False).mauve)
    results = []
    print("Edit Distance | Jaro Similarty | Cosine Similarity(BART) | PPL(GPT2-L) | MAUVE(to HWT) | MAUVE(b/a att) | BARTScore(to HWT) | BARTScore(b/a att) | BERTScore(to HWT) | BERTScore(b/a att) | PPL(Llama)")

    ed_str = f"{compute_average_std(edit_dists)[0]:.5g}+-{compute_average_std(edit_dists)[1]:.5g}"
    print(ed_str)
    results.append(["Edit Distance", ed_str])

    js_str = f"{compute_average_std(jaro_sim)[0]:.5g}+-{compute_average_std(jaro_sim)[1]:.5g}"
    print(js_str)
    results.append(["Jaro Similarty", js_str])

    if do_model_based: 
        cs_str = f"{compute_average_std(cos_sim)[0]:.5g}+-{compute_average_std(cos_sim)[1]:.5g}"
        print(cs_str)
        results.append(["Cosine Similarity(BART)", cs_str])

        ppl_str = f"{compute_average_std(gpt2l_ppl)[0]:.5g}+-{compute_average_std(gpt2l_ppl)[1]:.5g}"
        print(ppl_str)
        results.append(["PPL(GPT2-L)", ppl_str])

        if do_mauve:
            results.append(["MAUVE(to HWT)", f"{mauve_2h:.5g}"])
            print(f"{mauve_2h:.5g}")
            results.append(["MAUVE(b/a att)", f"{mauve_ab:.5g}"])
            print(f"{mauve_ab:.5g}")
        else: 
            results.append(["MAUVE(to HWT)", "-"])
            results.append(["MAUVE(b/a att)", "-"])
        
        # bart score
        bas2h_str = f"{compute_average_std(bart_ss_2h)[0]:.5g}+-{compute_average_std(bart_ss_2h)[1]:.5g}"
        print(bas2h_str)
        results.append(["BARTScore(to HWT)", bas2h_str])

        basab_str = f"{compute_average_std(bart_ss_ab)[0]:.5g}+-{compute_average_std(bart_ss_ab)[1]:.5g}"
        print(basab_str)
        results.append(["BARTScore(b/a att)", basab_str])

        bes2h_str = f"{compute_average_std(bert_ss_2h)[0]:.5g}+-{compute_average_std(bert_ss_2h)[1]:.5g}"
        print(bes2h_str)
        results.append(["BERTScore(to HWT)", bes2h_str])

        besab_str = f"{compute_average_std(bert_ss_ab)[0]:.5g}+-{compute_average_std(bert_ss_ab)[1]:.5g}"
        print(besab_str)
        results.append(["BERTScore(b/a att)", besab_str]) 

        ppl_llama_str = f"{compute_average_std(llama_ppl)[0]:.5g}+-{compute_average_std(llama_ppl)[1]:.5g}"
        print(ppl_llama_str)
        results.append(["PPL(Llama)", ppl_llama_str]) 

    return results      
