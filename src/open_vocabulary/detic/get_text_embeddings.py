import json
import re
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, 'third_party/CenterNet2/')

from detic.predictor import get_clip_embeddings


def get_text_embedding(caption: str = None, class_names: List[str] = None)-> Tuple[Any, List[List[float]]]:
    assert caption is not None or class_names is not None, "Both parameters are None!"

    if class_names is None:
        class_names = caption.split(',')
    text_embedding = get_clip_embeddings(class_names)  
    return class_names, text_embedding.detach().cpu().tolist()


def caption_preprocess(caption: str) -> List[str]:
    """
    For the text encoders that could not handle a sentence, but only a list of words.
    We preprocess the caption to extract all the unique words in the caption.

    Although the order may not matter, we preserve the order in which the words appear in the caption.
    For now, we only consider English words, which means non-English words, e.g., Chinese words, will be ignored.
    """
    word_pattern = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
    # Convert the caption to a string and lower-case it
    caption_str = str(caption).lower()
    
    # Find all alphabetical words
    words_found = word_pattern.findall(caption_str)
    
    # Remove duplicates but preserve the order in which they appear
    seen = set()
    unique_words = []
    for w in words_found:
        if w not in seen:
            seen.add(w)
            unique_words.append(w)
    return unique_words

    
def get_all_text_embeddings(label_des_df: pd.DataFrame) -> Dict[str, List[float]]:
    """
    Args:
        label_des_df: A pandas DataFrame. The 'idx' is the index of label. 
        The 'caption' is the description of the label.

    Returns:
        result: A dictionary with the 'idx' as the key. 
        The 'unique_words' are a list of unique words in the caption.
    """
    result = dict()
    for idx, caption in zip(label_des_df.iloc[:, 0], label_des_df.iloc[:, -1]):
        unique_words = caption_preprocess(caption)
        _, text_embedding = get_text_embedding(class_names=unique_words)
        result[idx] = {
            'idx': idx,
            'caption': caption,
            'words': unique_words,
            'embedding': text_embedding,
        }

    return result


def dump_text_embeddings(
    class_des_file: str = "rp_categories.csv",
    out_file: str = "rp_text_embeddings.json",
) -> None:
    """
    We pre-compute the text embeddings for all classes (captions) in the dataset.
    The text embeddings will be used as the classifier for the Detic.
    In this way, we can improve the efficiency by avoiding recomputing the embeddings for each inference.
    
    Args:
        class_des_file: A CSV file containing the description of each class.
        out_file: The output file name.
    """
    df = pd.read_csv(class_des_file)
    result = get_all_text_embeddings(df)

    with open(out_file, 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    dump_text_embeddings()
