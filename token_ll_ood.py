import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(0)
import numpy as np
np.random.seed(1)
import nltk
from tqdm import tqdm
from utils import load_csv 
from torch.utils.data import TensorDataset
from transformers import AutoModelForPreTraining, AutoTokenizer, AutoModelForCausalLM
OVERWRITE = 1
MAX_SEQ_LENGTH = 512
DEVICE = "cuda"
gen_ids   = ["gpt1", "gpt2sm", "gpt2lg", "llama13B", "gpt4"]
model_id = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def to_tensor(data):
    pad_token = 0
    labels = torch.stack([torch.tensor([int(d['label'])], dtype=torch.float32) for d in data]).squeeze()
    all_input_ids, all_attention_masks = [], []
    for d in data:
        inputs = tokenizer(d['sequence'])
        input_ids, attention_masks = inputs['input_ids'], inputs['attention_mask']
        padding_length = MAX_SEQ_LENGTH - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_masks = attention_masks + ([0] * padding_length)
        input_ids = input_ids[:MAX_SEQ_LENGTH]
        attention_masks = attention_masks[:MAX_SEQ_LENGTH]

        assert len(input_ids) == MAX_SEQ_LENGTH, "Error with input length {} vs {}".format(len(input_ids), MAX_SEQ_LENGTH)
        assert len(attention_masks) == MAX_SEQ_LENGTH, "Error with input length {} vs {}".format(len(attention_masks), MAX_SEQ_LENGTH) 

        all_input_ids.append(input_ids)    
        all_attention_masks.append(attention_masks)

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.int)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.int)
    dataset = TensorDataset(labels, all_input_ids, all_attention_masks)
    return dataset

def get_ll(text):
    with torch.no_grad():
        tokenized = model(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -model(**tokenized, labels=labels).loss.item()

def to_tokens_and_logprobs(input_texts, batch_size=32):
    if tokenizer.pad_token is None:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})
    encoded_input_texts = tokenizer(input_texts, add_special_tokens=False, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt", return_attention_mask=True).to(DEVICE)
    encoded_texts = encoded_input_texts["input_ids"]
    attn_masks = encoded_input_texts["attention_mask"]
    assert torch.all(
        torch.ge(attn_masks.sum(1), 2)
    ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    stds, means = [], []
    for start_index in range(0, len(encoded_texts), batch_size):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        probs = -torch.log(out_logits.softmax(dim=-1)).detach()

        # collect the probability of the generated token
        # we need to add a dummy dim in the end to make gather work
        gen_probs = torch.gather(probs, 2, encoded_batch[:, :, None]).squeeze(-1)

        std, mean = torch.std_mean(gen_probs, dim=1)
        
        stds.extend(std.tolist())
        means.extend(mean.tolist())

    assert len(stds) == len(means) and len(stds) == len(input_texts), "Output size does not match with orginal input."
    return stds, means

tot_means, tot_stds, tot_labels = [], [], []

# Machine text
for idx, gen_id in enumerate(['Human'] + gen_ids):
    if os.path.exists(f'll_figs/{gen_id}-{model_id.replace("/","@")}.pth') and not OVERWRITE:
        means, stds, labels = torch.load(f'll_figs/{gen_id}-{model_id.replace("/","@")}.pth')
        print("*Sucess Load from "+ f'll_figs/{gen_id}-{model_id.replace("/","@")}.pth')
    else:
        if gen_id == 'Human':
            # Human text
            dataset_path = {
                "test": f"multi_model_data/news/{gen_ids[1]}_test.csv"
            }
            test_dataset = load_csv(dataset_path["test"])
            data = [d['sequence'] for d in test_dataset if int(d['label'])==1]
            labels = ['Human'] * len(data)
        else:
            dataset_path = {
                "test": f"multi_model_data/news/{gen_id}_test.csv"
            }
            test_dataset = load_csv(dataset_path["test"])
            data = [d['sequence'] for d in test_dataset if int(d['label'])==0]
            labels = [gen_id] * len(data)
        # data_len = [len(d) for d in data]
        # data_flat = [item for sublist in data for item in sublist]
        # encodings = to_tensor(test_dataset)
        stds, means = to_tokens_and_logprobs(data)
        tot_stds.extend(stds)
        tot_means.extend(means)
        tot_labels.extend(labels)
        # from evaluate import load
        # perplexity = load("perplexity",  module_type= "measurement")
        # results = perplexity.compute(
        #     data=data_flat, 
        #     model_id=model_id,
        #     batch_size=32,
        #     device=DEVICE)['perplexities']
        # start_pos = 0
        # for idx, l in tqdm(enumerate(labels)):
        #     temp_ppl = results[start_pos: (start_pos+data_len[idx])]
        #     means.append(np.mean(temp_ppl))
        #     stds.append(np.std(temp_ppl))
        #     # if l==0 and (np.mean(temp_ppl)>50 or np.std(temp_ppl)>50):
        #     #     print(f"***Machine Sample {idx}: Mean {round(np.mean(temp_ppl),4)} SD {round(np.std(temp_ppl),4)}")
        #     #     print(test_dataset[idx])
        #     start_pos += data_len[idx]
        torch.save([means, stds, labels], f'll_figs/{gen_id}-{model_id.replace("/","@")}.pth')
    # tot_means.extend(means)
    # tot_stds.extend(stds)
    # tot_labels.extend(labels)
import pandas as pd
pd_data = pd.DataFrame({'x': tot_means, 'y': tot_stds, 'label': tot_labels})
pd_data = pd_data.sample(frac=1).reset_index(drop=True)
    # std_portions = [[], []]
    # std_portions[0] = [s/m for m,s in zip(means[0], stds[0])]
    # std_portions[1] = [s/m for m,s in zip(means[1], stds[1])]
# condition = (pd_data['x'] > 400) | (pd_data['y'] > 400)
# pd_data = pd_data[~condition]
pd_data['sum'] = pd_data['x'] + pd_data['y']
threshold = pd_data['sum'].quantile(0.95)
pd_data = pd_data[pd_data['sum'] <= threshold]
pd_data['y/x'] = pd_data['y'] / pd_data['x'] 
sns.scatterplot(x='x', y='y/x', hue='label', data=pd_data, palette=sns.color_palette("Paired"))
label2color = {}
for l,c in zip(pd_data['label'].unique().tolist(), sns.color_palette("Paired")):
    #Make it deeper
    label2color[l] = (c[0] / 2, c[1] / 2, c[2] / 2)
 

# class_means = pd_data.groupby('label').mean()
class_stats = pd_data.groupby('label').agg({'x': ['mean', 'std'], 'y/x': ['mean', 'std']}).reset_index()
# plt.scatter(class_means['x'], class_means['y'], marker='X', color='black', s=100)
output_txt = f'll_figs/{"&".join(gen_ids)}-{model_id.replace("/","@")}.txt'
with open(output_txt, 'w') as file:
    for index, row in class_stats.iterrows():
        # plt.scatter(row['x'], row['y'], marker='x', color='black', s=100)
        file.write(class_stats.label[index]+'\n')
        x_mean, x_std = row['x']['mean'], row['x']['std']
        y_mean, y_std = row['y/x']['mean'], row['y/x']['std']
        file.write(f"Mean [{round(x_mean-x_std, 2)}, {round(x_mean+x_std, 2)}] \n")
        file.write(f"Std  [{round(y_mean-y_std, 2)}, {round(y_mean+y_std, 2)}] \n")
        plt.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, marker='x', markersize=10, capsize=4, color=label2color[class_stats.label[index]])
        plt.annotate(class_stats.label[index], (x_mean, y_mean), textcoords="data", 
                     #xytext=(30, 10), 
                     ha='center', annotation_clip=True,
                     fontsize=5, fontweight='bold')
    #
    # class_bounds = pd_data.groupby('label').quantile([0.025, 0.975]).unstack(level=1)

    # for i, row in class_bounds.iterrows():
    #     plt.plot([row['x', 0.025], row['x', 0.975]],
    #              [row['y', 0.025], row['y', 0.975]], 'k--')

    plt.xlabel('Mean of -log_likelihood')
    plt.xlim(8, 13)
    # plt.ylabel('SD')
    plt.ylim(0.1, 0.4)
    plt.ylabel('SD/Mean')
    # plt.ylim(0, 2)    
    plt.legend()
    plt.title(model_id + " detector")
    plt.savefig(f'll_figs/{"&".join(gen_ids)}-{model_id.replace("/","@")}.png', dpi=600) # gen-detect

# import numpy as np
# output_txt = f'll_figs/{gen_id}-{model_id.replace("/","@")}.txt'
# with open(output_txt, 'w') as file:
#     file.write("Machine\n")
#     mean = np.mean(aggregate_ppl[0])
#     median = np.median(aggregate_ppl[0])
#     std = np.std(aggregate_ppl[0])
#     file.write("  Mean  : " + str(mean) +'\n')
#     file.write("  Median: " + str(median) + '\n')
#     file.write("  STD   : " + str(std) + '\n')
#     file.write("Human\n")
#     mean = np.mean(aggregate_ppl[1])
#     median = np.median(aggregate_ppl[1])
#     std = np.std(aggregate_ppl[1])
#     file.write("  Mean  : " + str(mean) +'\n')
#     file.write("  Median: " + str(median) + '\n')
#     file.write("  STD   : " + str(std) + '\n') 
#     file.write("Difference\n")
#     file.write("  Mean  : " + str(np.mean(aggregate_ppl[0])-np.mean(aggregate_ppl[1])) +'\n')
#     file.write("  Median: " + str(np.median(aggregate_ppl[0])-np.median(aggregate_ppl[1])) + '\n')
#     file.write("  STD   : " + str(np.std(aggregate_ppl[0])-np.std(aggregate_ppl[1])) + '\n')      
# from datasets import load_dataset

# test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# encodings = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")


# max_length = model.config.n_positions # model max length 1024 for gpt
# stride = 512
# seq_len = encodings.input_ids.size(1) # length of dataset

# nlls = []
# prev_end_loc = 0
# for begin_loc in tqdm(range(0, seq_len, stride)):
#     end_loc = min(begin_loc + max_length, seq_len)
#     trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
#     input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
#     target_ids = input_ids.clone()
#     target_ids[:, :-trg_len] = -100

#     with torch.no_grad():
#         outputs = model(input_ids, labels=target_ids)

#         # loss is calculated using CrossEntropyLoss which averages over valid labels
#         # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
#         # to the left by 1.
#         neg_log_likelihood = outputs.loss

#     nlls.append(neg_log_likelihood)

#     prev_end_loc = end_loc
#     if end_loc == seq_len:
#         break

# ppl = torch.exp(torch.stack(nlls).mean())