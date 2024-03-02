import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau


def eval_cog(cog_path: str, tokenizer, special: str | None):
    cog_data = pd.read_csv(cog_path)
    cog_data = cog_data.dropna()
    words = cog_data[cog_data["lexicality"] == "W"]
    nonwords = cog_data[cog_data["lexicality"] == "N"]

    datasets = {"words": words, "nonwords": nonwords}
    all_results = {}
    avg_corr = 0
    for category, dataset in datasets.items():
        # measurements
        words = list(dataset["spelling"])
        rts = list(dataset["rt"])
        accs = list(dataset["accuracy"])

        # splits in model output
        tokens = list([tokenizer.tokenize(word) for word in words])
        wordiness = [1 - (len(tokens[i]) / len(str(words[i]))) for i in range(len(dataset))]

        # correlation
        corr1, p1 = pearsonr(wordiness, rts)
        corr2, p2 = pearsonr(wordiness, accs)

        category_results = {category + "_chunkability_rts": corr1, category + "_chunkability_accs": corr2}
        avg_corr += abs(corr1)
        avg_corr += abs(corr2)
        all_results.update(category_results)
    all_results["cog_score"] = avg_corr/4
    return all_results
