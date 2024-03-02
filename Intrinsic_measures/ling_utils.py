from typing import List


# Function to get_boundaries from a tokenization
def get_boundaries(tokenization: List[str]) -> List[int]:
    boundaries = [len(''.join(tokenization[:i])) for i in range(1, len(tokenization))]
    return boundaries


def get_seg_coverage(x, tokenizer, key_in_df, get_gstandard,
                     special: str) -> dict[str:float]:
    tps = fps = fns = length = count = 0
    for _, row in x.iterrows():
        # Gold standard morphological segmentation from the dataset
        gstandard = get_gstandard(row)
        gstandard[0] = "Ġ" + gstandard[0]
        if "".join(gstandard) not in tokenizer.get_vocab():
            if special == "##":
                gstandard = list(map(lambda tok: "##" + tok, gstandard))
                gstandard[0] = gstandard[0][2:]
            if all(
                    token in tokenizer.get_vocab() for token in gstandard):
                count += 1
                # Tokenise the compound with the given tokeniser
                y = [x for x in tokenizer.tokenize(row[key_in_df])]
                # Get the boundaries for the gold standard and the tokeniser
                gstandard_boundaries = get_boundaries(gstandard)
                y_boundaries = get_boundaries(y)
                fn = 0
                for i in y_boundaries:
                    if i in gstandard_boundaries:
                        # True positives are those appearing in both generated and reference
                        tps += 1
                    else:
                        # False positives are those appearing in the generated but not the reference
                        fps += 1
                for i in gstandard_boundaries:
                    if i not in y_boundaries:
                        # False negatives are those appearing in the reference but not the generated
                        fn += 1
                fns += fn
                length += len(y)
    f1 = tps / (tps + 0.5 * (fps + fns))
    return {"f1": f1}