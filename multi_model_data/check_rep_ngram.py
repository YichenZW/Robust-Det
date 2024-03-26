import numpy as np
def rep_ngram(sen_lis, num_gram=4):
    rep_lis = []
    for sen in sen_lis:
        uniq_ngram, all_ngram = {}, []
        for i in range(0, len(sen) - num_gram + 1):
            tt = ' '.join(sen[i:i + num_gram])
            if not tt in uniq_ngram: uniq_ngram[tt] = True
            all_ngram.append(tt)
        if len(all_ngram) == 0:
            print(f'warning: len(all_ngram) is 0!!! skipping... sample: {str(sen)}')
            continue
        rep = 1.0 - len(uniq_ngram) * 1.0 / len(all_ngram)
        rep_lis.append(rep)
    return np.mean(rep_lis)
def load_csv(path):
    import csv
    data = []
    print(f"*Loading data from {path}.")
    with open(path, 'r') as file:
        csv_reader = csv.DictReader(file, delimiter='|')
        for row in csv_reader:
            data.append(row)
    print("*Success. {} Lines Loaded.".format(len(data)))
    return data

attack_path = "multi_model_data/news_gptj_t1.5/gptj_watermark_t1.5_test.csv"
dataset = load_csv(attack_path)
mgts = [p['sequence'] for p in dataset if p['label']=='0']
print(attack_path, ":", rep_ngram(mgts))
