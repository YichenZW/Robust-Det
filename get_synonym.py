from nltk.corpus import wordnet
import re
import random

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

pronouns = ["I", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them", "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs", "this", "that", "these", "those", "who", "whom", "whose", "which", "what", "myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves"]
other_function_words = ["a", "an", "the", "and", "but", "or", "nor", "for", "yet", "so", "as", "if", "is", "are", "be", "was", "were", "being", "been"]
stop_words = pronouns + other_function_words
def synonym_subst(pct_words_masked, text, mode):
    words = re.findall(r"[\w']+|[.,!?;]", text)
    subst_num = round(pct_words_masked * len(words))
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
            print(f"{words[subject_word_idx]} -> {synonym}")
            words[subject_word_idx] = synonym
            break
    substed_text = " ".join(words)
    return substed_text
