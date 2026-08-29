from typing import List, Union
import tokenization_scorer


def encode_corpus(corpus: List[str], tokenized_corpus: List[List[str]]) -> dict[str:float]:
    res = {}
    num_of_tokens = sum([len(tokenized_sentence) for tokenized_sentence in tokenized_corpus])
    num_of_words = sum(len(sentence.split(" ")) for sentence in corpus)
    res["fertility"] = num_of_tokens / num_of_words
    return res


def entropy_scores(tokenized_corpus: List[List[str]]) -> dict[str:float]:
    res = {}
    res["entropy_score"] = tokenization_scorer.score(tokenized_corpus,power=2.5)
    return res
