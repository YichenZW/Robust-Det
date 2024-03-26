import os
import torch
from tqdm import tqdm
from utils import load_csv 
from torch.utils.data import TensorDataset
from transformers import AutoModelForPreTraining, AutoTokenizer, AutoModelForCausalLM
OVERWRITE = True
MAX_SEQ_LENGTH = 512
device = "cuda"
gen_id   = "gpt2lg"
model_id = "transfo-xl-wt103"

# model = AutoModelForCausalLM.from_pretrained(model_id, is_decoder=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, bos_token='<BOS>')

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

if os.path.exists(f'figs/{gen_id}-{model_id.replace("/","@")}.pth') and not OVERWRITE:
    aggregate_ppl = torch.load(f'figs/{gen_id}-{model_id.replace("/","@")}.pth')
    print("*Sucess Load from "+ f'figs/{gen_id}-{model_id.replace("/","@")}.pth')
else:
    DATASET_PATH = {
        "test": f"multi_model_data/news/{gen_id}_test.csv"
    }
    test_dataset = load_csv(DATASET_PATH["test"])
    data = [d['sequence'] for d in test_dataset]
    labels = [d['label'] for d in test_dataset]
    # encodings = to_tensor(test_dataset)

    from evaluate import load
    perplexity = load("perplexity",  module_type= "measurement")
    results = perplexity.compute(
        data=data, 
        model_id=model_id,
        batch_size=16,
        device=device)
    aggregate_ppl = [[], []]
    for l, ppl in zip(labels, results['perplexities']):
        aggregate_ppl[int(l)].append(ppl)
    torch.save(aggregate_ppl, f'figs/{gen_id}-{model_id.replace("/","@")}.pth')

import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(aggregate_ppl[0], stat='probability', binwidth=1, binrange=(0, 50), kde=True, color='royalblue', label="Machine")
sns.histplot(aggregate_ppl[1], stat='probability', binwidth=1, binrange=(0, 50), kde=True, color='darkorange', label="Human")
plt.xlabel('Value')
plt.xlim(0, 60)
plt.ylabel('Frequency')
plt.title('Distribution Histogram')
plt.legend()
plt.savefig(f'figs/{gen_id}-{model_id.replace("/","@")}.png', dpi=600) # gen-detect

import numpy as np
output_txt = f'figs/{gen_id}-{model_id.replace("/","@")}.txt'
with open(output_txt, 'w') as file:
    file.write("Machine\n")
    mean = np.mean(aggregate_ppl[0])
    median = np.median(aggregate_ppl[0])
    std = np.std(aggregate_ppl[0])
    file.write("  Mean  : " + str(mean) +'\n')
    file.write("  Median: " + str(median) + '\n')
    file.write("  STD   : " + str(std) + '\n')
    file.write("Human\n")
    mean = np.mean(aggregate_ppl[1])
    median = np.median(aggregate_ppl[1])
    std = np.std(aggregate_ppl[1])
    file.write("  Mean  : " + str(mean) +'\n')
    file.write("  Median: " + str(median) + '\n')
    file.write("  STD   : " + str(std) + '\n') 
    file.write("Difference\n")
    file.write("  Mean  : " + str(np.mean(aggregate_ppl[0])-np.mean(aggregate_ppl[1])) +'\n')
    file.write("  Median: " + str(np.median(aggregate_ppl[0])-np.median(aggregate_ppl[1])) + '\n')
    file.write("  STD   : " + str(np.std(aggregate_ppl[0])-np.std(aggregate_ppl[1])) + '\n')      
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