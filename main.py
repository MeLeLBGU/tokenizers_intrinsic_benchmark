import os.path, argparse
import utils
from tqdm import tqdm
from Intrinsic_measures import static, ling, human_comp, compare
import pandas as pd
from const import *
from benchmark_objects import BenchmarkTokenizer


def load_args():
    parser = argparse.ArgumentParser(description="Sub-word tokenizers intrinsic benchmark")
    parser.add_argument("--tokenizers", default="tokenizers.txt",
                        help="A path to a txt file containing paths to tokenizers tokenizers")
    parser.add_argument("--compare",help="A flag for comparing the segmentation difference between the tokenizers" ,action="store_false")
    args = vars(parser.parse_args())
    if not args["tokenizers"]:
        parser.error("You must specify the tokenizers path")
    return args


def run_static(tokenizer, corpus):
    metrics = {}
    metrics.update(static.encode_corpus(tokenizer, corpus))
    metrics.update(static.entropy_scores(tokenizer, corpus))
    return metrics


def run_ling(tokenizer, all_tokenizers, vocab, special):
    metrics = {}
    metrics.update(
        ling.combined_coverage(COMBINED, tokenizer, special))
    return metrics


def run_human(tokenizer, special, verbose=False):
    metrics = {}
    metrics.update(human_comp.eval_cog(EN, tokenizer, special))
    return metrics


def run_comp(all_tokenizers, corpus,special):
    compare.segmentation_diff(all_tokenizers[0], all_tokenizers[1:], corpus,special)


def eval_tokenizer(tokenizer, all_tokenizers, special, compare):
    # convert the corpus to a list of strings
    corpus = utils.corpus_to_list(MINIPILE_TEST)
    vocab = tokenizer.get_vocab()
    metrics = {"type": tokenizer.get_type()}

    # Static metrics
    metrics.update(run_static(tokenizer, corpus))

    # Linguistic metrics
    metrics.update(run_ling(tokenizer, all_tokenizers, vocab, special))

    # human metrics
    metrics.update(run_human(tokenizer, special))

    # comparative measures
    if compare and tokenizer == all_tokenizers[0]:
        # This function doesn't return a value and just prints the segmentation difference once
        # This counts on the fact that the vocabulary is the same for all tokenizers
        # And that the default inference is the first tokenizer
        run_comp(all_tokenizers, corpus,special)

    return metrics


def main():
    args = load_args()
    names = []
    df = {"tokenizer": names}
    metrics = []
    with open(args['tokenizers'], 'r') as vocabs_file:
        paths = [path.strip() for path in vocabs_file.readlines()]
        tokenizers = [BenchmarkTokenizer(path) for path in paths]
        for path, tokenizer in tqdm(zip(paths, tokenizers)):
            file_name = os.path.basename(path)
            special = "##" if (file_name.startswith("wordpiece") or file_name.startswith("flota_wordpiece") or \
                               file_name.startswith("suffix_wordpiece")) else "Ġ"
            try:
                results = eval_tokenizer(tokenizer, tokenizers, special,args['compare'])
                if not metrics:
                    metrics = list(results.keys())
                    for metric in metrics:
                        df[metric] = []
                names.append(os.path.basename(path).rstrip(".json"))
                for metric in metrics:
                    df[metric].append(results[metric])
            except Exception as e:
                print(f"An error occurred on {path.strip()}: {e}")

    df = pd.DataFrame(df).round(4)
    df.to_csv('output.csv', index=False)


main()
