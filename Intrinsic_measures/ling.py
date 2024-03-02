import pandas as pd
from Intrinsic_measures.ling_utils import get_seg_coverage


def combined_coverage(combined_path: str, tokenizer, special):
    df = pd.read_csv(combined_path, sep=",")
    get_gstandard = lambda row: eval(row['Gold_standard_segmentation'])
    datasets = ["Ladec", "MorphoLex", "MorphyNet", "Dago_Bert", "UniMorph", "UnBlend", "CompoundPiece"]
    coverage = {}
    avg_f1 = 0
    for dataset in datasets:
        curr_coverage = get_seg_coverage(df.loc[df['Origin'] == dataset], tokenizer,
                                         'Word',
                                         get_gstandard, special)
        avg_f1 += curr_coverage['f1']
        curr_coverage = {dataset + "_" + key: val for key, val in curr_coverage.items()}
        coverage.update(curr_coverage)
    coverage["avg_f1"] = avg_f1 / len(datasets)
    return coverage
