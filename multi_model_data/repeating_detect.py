import re
import os
import pandas as pd

def find_repeating_subsequences(text, min_len=20):
    """
    Finds repeating substrings of length at least min_len in text.
    
    Args:
    - text (str): The text to be analyzed.
    - min_len (int): The minimum length of repeating substrings to be considered.
    
    Returns:
    List[str]: A list of repeating substrings.
    """
    seq_dict = {}
    if type(text) is not str:
        return 0
    for i in range(len(text) - min_len + 1):
        subseq = text[i:i + min_len]
        if subseq in seq_dict:
            seq_dict[subseq].append(i)
        else:
            seq_dict[subseq] = [i]
    
    # Filtering subsequences that appear more than once.
    repeating_subseqs = {k: v for k, v in seq_dict.items() if len(v) > 1}
    repeating_subseqs = list(repeating_subseqs.items())
    if repeating_subseqs!=[]:
        pre_pair = repeating_subseqs[0]
        for pair in repeating_subseqs[1:]:
            if pre_pair[0][1:] == pair[0][:-1]:
                repeating_subseqs.remove(pre_pair)
            pre_pair = pair
    return repeating_subseqs

def evaluate_text(text):
    """
    Evaluates text for repetition and presents findings.
    
    Args:
    - text (str): The text to be analyzed.
    
    Returns:
    None: Prints findings.
    """
    repeating_subseqs = find_repeating_subsequences(text, min_len=40)
    
    if not repeating_subseqs:
        print("No repeating substrings found.")
        return 0
    else:
        print("Repeating substrings found:")
        for subseq, indices in repeating_subseqs:
            print(f"  '{subseq}' at indices {', '.join(map(str, indices))}")
        return len(repeating_subseqs)

if __name__ == "__main__":
    TESTSET_PATH = "multi_model_data/news_gptj_t1.5/gptj_watermark_t1.5_test.csv"

    df = pd.read_csv(TESTSET_PATH, sep="|")
    print("*Loaded from", TESTSET_PATH)

    tot = 0
    for index, d in df.iterrows():
        if d["label"] == 1: # human written texts
            continue
        res = evaluate_text(d["sequence"])
        tot += res
        if res:
            print(f"*ID{index}: {res}")
    print("*Loaded from", TESTSET_PATH)
    print(f"*Total: {tot}")
