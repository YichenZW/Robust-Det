import openai
import time
import random
random.seed(0)
from tqdm import tqdm
from utils import load_csv
openai.api_key = "your/openai/key"

def gpt4_eval(texts_ori, texts_att):
    # semantically same
    # which is low quality in writing.
    sems, qs = [], [] # sem higher=more similar, q higher=att q higher
    for t_ori, t_att in tqdm(zip(texts_ori, texts_att)):
        do_reverse = random.randint(0, 1)
        if not do_reverse:
            messages = [{"role": "user", "content": f"Read the following two passages: \n\n A: {t_ori} \n\n B: {t_att} \n\n"} ]
        else:
            messages = [{"role": "user", "content": f"Read the following two passages: \n\n A: {t_att} \n\n B: {t_ori} \n\n"} ]            
        messages.append({"role": "user", "content": "1) Are they the same in semantics? Answer Yes or No."})
        messages.append({"role": "user", "content": "2) Which one is relatively low quality in writing? Answer A or B."})
        messages.append({"role": "user", "content": "Respond in this format: (Yes/No)|(A/B)"})
        response = -1
        while response == -1:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-0613",
                    messages=messages,
                    temperature=0
                )
            except Exception as e:
                print("GPT-API Error:", e)
                time.sleep(5)
                continue
        ans = response['choices'][0]['message']['content'].split("|")
        if not do_reverse:
            if ans[0] == "Yes":
                sems.append(1)
            elif ans[0] == "No":
                sems.append(0)
            else:
                print("Informal respone: ", ans[0])
            if ans[1] == "A":
                qs.append(1)
            elif ans[1] == "B":
                qs.append(0)
            else:
                print("Informal respone: ", ans[1])
        else:
            if ans[0] == "Yes":
                sems.append(1)
            elif ans[0] == "No":
                sems.append(0)
            else:
                print("Informal respone: ", ans[0])
            if ans[1] == "A":
                qs.append(0)
            elif ans[1] == "B":
                qs.append(1)
            else:
                print("Informal respone: ", ans[1])            
    print(f"GPT-4 Similarity ^: {(sum(sems)/len(sems)):.5f}")
    print(f"GPT-4 Quality ^:    {(sum(qs)/len(qs)):.5f}")
        
def main():
    TEST_ORI = "multi_model_data/news/gpt2md_test.csv"
    TEST_ATT = "multi_model_data/news/gpt2md_test.typo_trans0.05_att.csv"
    test_dataset_ori = load_csv(TEST_ORI)
    test_dataset_att = load_csv(TEST_ATT)
    test_dataset_att_mgt = [d for d in test_dataset_att if d['label']=='0']
    test_dataset_ori_mgt = [d for d in test_dataset_ori if d['label']=='0']
    gpt4_eval(test_dataset_ori_mgt, test_dataset_att_mgt)

if __name__ == "__main__":
    main()