# Stumbling Blocks: Stress Testing the Robustness of Machine-Generated Text Detectors Under Attacks

![Image text](https://github.com/YichenZW/Robust-Det/blob/main/img-illust/GithubIllust.jpg)

This repository includes the code implementation of the paper **Stumbling Blocks: Stress Testing the Robustness of Machine-Generated Text Detectors Under Attacks** ([arxiv](https://arxiv.org/abs/2402.11638)) by *Yichen Wang, Shangbin Feng, Abe Bohan Hou, Xiao Pu, Chao Shen, Xiaoming Liu, and Yulia Tsvetkov,* and *Tianxing He,* mainly at Paul G. Allen School of CSE, University of Washington. We comprehensively reveal that almost none of the existing machine-generated text detectors remain robust under all the attacks, and all detectors exhibit different loopholes. Further, we investigate the reasons behind these defects and propose initial out-of-the-box patches to improve robustness. 
The code can be used for detection research as a robustness evaluation benchmark.

## A. Installation

(1) Install Python 3.9.18 and PyTorch 2.0.1. (slightly older or newer versions are probably also fine for both).

(2) Install other packages `pip install -r requirements.txt`.

## B. Dataset Generation

**You can directly load our dataset from [huggingface link].** Save data in to `multi_model_data/` path.

---

**a) Direct generation:**

```shell
python multi_model_data/gen_dataset.py --gen_model_name Llama2-7b-hf --top_p 0.96 --temp 1.0 --do_sample True  
```

Other arguments:

- `--top_p` `--temp` and `--rp` can be customized, a detailed discussion in paper Appendix A.1 (page 16).
  
- Note that if you need to use OpenAI models, please set `"OPENAI_API_KEY"` in advance.
  

**b) Watermarked generation:**

An example of watermarked generation.

```
python watermark/gen_watermark.py --cache_dir cache --gen_model_name Llama2-7b-hf --domain news_Llama2-7b-hf_t1.5
```

**c) Evaluating repetition:**

For reasonable testing, we need to avoid repetition in generation.

- `check_rep_ngram.py` is for checking repetition by comparing n-grams.
  
- `repeating_detect.py` is also used to check repetition by comparing sub-strings.
  

If the generated text is too repetitive, we suggest setting the temperature and repetition penalty to be larger.

**d) Prompt generation**: get a dataset of the prompt for further attacked generation. It is used for *prompt paraphrasing attack (Section 6.4(1))*

- `gen_prompt.py` can save an additional file for the prompts used.

## C. Attack

**a) Post-generation attacks:**

```
python post_attack.py --dataset_name <...> --generate_model_name <...> --attack_method <...> --pct_words_masked <...>
```

Post-generation attacks include:

- Typo Insertion (Sec. 6.2(1)). attack_method = `typo_(mix/trans/subst/delet/insert)`, use `--pct_words_masked` to set the percentage of words with typo. One typo per selected word.
  
  E.g., `python post_attack.py --dataset_name news_gptj_t1.5 --generate_model_name gptj --attack_method typo_mixed --pct_words_masked 0.05`
  
- Homoglyph Alteration (Sec. 6.2(2)) attack_method = `homo_(ECES/ICES)`. We recommend using ECES.`--pct_words_masked` to set the attacked word percentage.
  
- Format Character Editing (Sec. 6.2(3)) attack_method = `form_(shift/zero-sp)`. Shifting is edited at sentence boundaries (`--pct_words_masked` to set the sentence percentage), and zero-width space is added at any position (`--pct_words_masked` to set the attacked word percentage.).
  
- Synonyms Substitution (Sec. 6.3(1)) attack_method = `word_subst_(modelfree/modelbase)`. For the model-free substitution, we replace the selected words with their synonyms retrieved from a static dictionary WordNet. In the model-based method, we use T5-large to select the words to be substituted and prompt LlaMA to get the synonyms given the context.
  
- Span Perturbation (Sec. 6.3(2)) attack_method = `ptb`. `--pct_words_masked` to set the attacked span percentage. Additionally, `--mask_filling_model_name` is to set masking model, `--span_length` set the span length, and `--top_p` `--top_k` `--mask_top_p` are configs for the mask-filling sampling strategy.
  
- Inner-Sentence Paraphrase (Sec. 6.3(3)). attack_method = `pegasus`. `--pct_words_masked` to set the paraphrased sentence percentage. Change `--mask_top_p` for the paraphraser strategy.
  
- Inter-Sentence Paraphrase (Sec. 6.3(4)). attack_method = `dipper`. Use `--lex_diversity` and `order_diversity` to control the perturbation level. The values available are 0, 20, 40, 60, 80, and 100. For example, `python post_attack.py --dataset_name news_gptj_t1.5 --generate_model_name gptj --attack_method dipper --lex_diversity 40 order_diversity 40`. Do not use `--pct_words_masked`.
  
  *For all attacks here, set `--watermark` to True if you want to attack on the watermarked dataset.*
  

**b) Pre-generation attacks:**

```
python gen_attack.py --attack_method <...> --attack_args <...>
```

Additional arguments:

`--domain`: i.e., the name of the dataset folder.

`--model_list` and `--gen_model_name`: generator model.

`--top_p``--temp``--rp``--do_sample` for sampling strategy.

An example commanding in our paper is:

```
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \ 
python gen_attack.py --domain news_gptj_t1.5 --model_list EleutherAI/gpt-j-6b --gen_model_name gptj --temp 1.5  
```

Pre-generation attacks include:

- In-Context Learning (Sec. 6.4(2)). attack_method = `icl`.
- Character-Substituted Generation (Sec. 6.4(3)). attack_method = `csgen`.

*for the above two attacks, we suggest using GPT-4 for better generation quality.*

- Prompt Paraphrasing (Sec. 6.4(1)). First, run `gen_prompt.py` to get the raw prompts. Then, run `attacks/para_prompt.py` to paraphrase the prompts. Finally, run `attacks/para_prompt_based_gen.py` to generate full texts based on the prompt.

For the watermarked generation, the schema is quite different. Use `watermark/gen_watermark_prompt.py`. `TESTSET_PATH` is to specify the prompt dataset.

**c) On-generation attacks:**

- Emoji Co-Generation (Sec. 6.5(2)). Commands for example, `HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python gen_attack.py --attack_method emoji --attack_args 0.75`.`--attack_args` controls the probability of emoji insertion.
  
- Typo Co-Generation (Sec. 6.5(1)). Commands for example, `HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python gen_attack.py --attack_method typo-cogen --attack_args 0.75`.`--attack_args` controls the probability of substition applied. The default substitution rule here is to change 'c's into 'k's and 'k's into 'c's. You can customize any by changing `SUBST_RULE`.
  

*For the watermarked generation, the schema is quite different. Use `watermark/gen_watermark_cogen.py`e.g., `python watermark/gen_watermark_cogen.py --attack emoji --attack_ratio 0.06`*

## D. Detection

Run the unattacked setting, command example:

```
python detect/run.py --output_name main --base_model_name gpt-j --mask_filling_model_name t5-3b --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset multi --dataset_key news_gptj_t1.5 --generate_model_name gptj --gpu_id 0 --do_attack False
```

Run the attacked setting, command example:

```
python detect/run.py --output_name main --base_model_name gpt-j --mask_filling_model_name t5-3b --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset multi --dataset_key news_gptj_t1.5 --generate_model_name gptj --gpu_id 0 --do_attack True --attack_method dipper --att_args _L60O60
```

The result will be saved into an Excel form. If you need detailed logs, you are also recommended to use this shell script we wrote.

```shell
base_id="gpt-j"; 
base_name="gptj";
gen_id="gptj";
do_attack="True";
attack_method="dipper"; 
att_args="_L60O60"; 
gpu_id="0";
log_file="logs/${gen_id}/${gen_id}@${base_name}_${attack_method}${att_args}.log";
echo "$log_file";
export TIKTOKEN_CACHE_DIR="";
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \ 
python detect/run.py --output_name main --base_model_name "$base_id" --mask_filling_model_name t5-3b --n_perturbation_list 1,10 --n_samples 500 --pct_words_masked 0.3 --span_length 2 --dataset multi --dataset_key news_gptj_t1.5 --generate_model_name "$gen_id" --gpu_id "$gpu_id" --do_attack "$do_attack" --attack_method "$attack_method" --attack_args "$att_args"
```

The `attack_method` should match the name of the attack in the previous section. And the `att_args` here is `_<attack_args>` of the previous section (If no args are required, left ""; If it is a Dipper attack, use format `_L<...>O<...>`).

The detection code is based on DetectGPT.

### Watermark detection:

A command example:

```
python watermark/detect_watermark.py --attack emoji-cogen_0.06
```

The `--attack` is `<attack_name>_<attack_args>` you used to generate the attack. It is better just to check the filename of the attacked dataset directly.

The results will also be recorded in an Excel.

### Fine-tuned detector

If you want to test the fine-tuned detector, you need to run `classifier.py` to fine-tune a classifier first. For example,

```
python classifier.py --dataset_name news_gpt-4_t0.7 --generate_model_name gpt-4 --detect_model_name microsoft/deberta-v3-large
```

The model will be saved into `models/`. Then it will be tested if you run `detect/run.py`.

## E. Budget

Get the budget of attack by comparing the attacked dataset (`testset_att`) with the unattacked dataset (`--testset_ori`). The command example used for our paper is:

```shell
# for unwatermarked datasets 
python static_and_finetuned.py --testset_att multi_model_data/news_gptj_t1.5/gptj_test.typo_insert0.2_att.csv 
# for watermarked datasets
python static_and_finetuned.py --testset_ori multi_model_data/news_gptj_watermark/gptj_test.csv \
--testset_att multi_model_data/news_gptj_watermark/gptj_test.typo_insert0.2_att.csv \
--watermark
```

You can set `not_include_model_based` to `True` if you only want to run statistic metrics (e.g., distance ...).

The result will be saved in an Excel form.

## F. Patches

For the patched detector that we proposed in Sec. 6.2.3, run `detect/run_with_filter.py`. The arguments are the same as those we mentioned afore.

**If you have any questions, please feel free to ping us!**

## G. Citation

```
@article{wang2024stumbling,
  title={Stumbling Blocks: Stress Testing the Robustness of Machine-Generated Text Detectors Under Attacks},
  author={Wang, Yichen and Feng, Shangbin and Hou, Abe Bohan and Pu, Xiao and Shen, Chao and Liu, Xiaoming and Tsvetkov, Yulia and He, Tianxing},
  journal={arXiv preprint arXiv:2402.11638},
  year={2024}
}
```

_The work is done while Yichen Wang is interning at CSE University of Washington from Xi’an Jiaotong University. Other institutions involved are Johns Hopkins University (Abe Bohan Hou) and Peking University (Xiao Pu)._
