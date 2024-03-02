def segmentation_diff(default_tokenizer, others, corpus,special):
    corpus_str = "".join(corpus)
    splitted_text = corpus_str.split()
    default_tokenized_corpus = [default_tokenizer.tokenize(pre_token) for pre_token in splitted_text]
    tokenized_corpuses = [[tokenizer.tokenize(pre_token) for pre_token in splitted_text] for tokenizer in others]
    for i, tokenized_corpus in enumerate(tokenized_corpuses):
        tokenizer = others[i]
        diff = total = 0
        for default_tokenization, tokenization in zip(default_tokenized_corpus, tokenized_corpus):
            if default_tokenization != tokenization:
                if special == "##" and "equal" in tokenizer.get_type():
                    default_tokenization = list(map(lambda tok: "##" + tok if not tok.startswith("##") else tok,default_tokenization))
                    default_tokenization[0] = default_tokenization[0][2:]
                if default_tokenization != tokenization:
                    diff += 1
            total += 1
        print(f"for tokenizer {tokenizer.get_type()} the diff is {diff / total}")