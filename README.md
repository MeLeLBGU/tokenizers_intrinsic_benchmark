# tokenizers_intrinsic_benchmark

Code for paper "Greed is All You Need: An Evaluation of Tokenizer Inference Methods" (Link TBA)

## Requirements
Python packages are listed in requirements.txt.
This code does not require GPU\TPU.

## Notes
The benchmark supports tokenizers which are serialized into a HuggingFace json format.
In addition, we've added support for some custom inference methods (greedy longest suffix, greedy lognest token, etc.).
The json files we've used for the paper will be added soon as examples.

## Resources
The resources we've used for the evaluation are in the resources folder.

Resource | Reference 
| ------------- | ------------- |
LADEC | [paper](https://www.semanticscholar.org/paper/LADEC%3A-The-Large-Database-of-English-Compounds-Gagn%C3%A9-Spalding/7da138d704ef0fc055825fa132f5c452ed3fb52a)
MorphoLex | [paper](https://www.semanticscholar.org/paper/MorphoLex%3A-A-derivational-morphological-database-S%C3%A1nchez-Guti%C3%A9rrez-Mailhot/3cea3a3eb5b83612a7f8da49fde0d7244058ee06)
MorphyNet | [paper](https://aclanthology.org/2021.sigmorphon-1.5/)
DagoBert | [paper](https://aclanthology.org/2020.emnlp-main.316/)
UniMorph | [paper](https://aclanthology.org/2022.lrec-1.89/)
UnBlend | [paper](https://aclanthology.org/2020.findings-emnlp.138/)
CompoundPiece | [paper](https://aclanthology.org/2023.emnlp-main.24/)
Cognitive data | [paper](https://aclanthology.org/2023.emnlp-main.272/)
tokenization-scorer | [paper](https://aclanthology.org/2023.acl-long.284/)

## Execution
Execute `main.py` from its working directory.

arguments:
```	
	--tokenizers: a path to a txt file containing paths to tokenizers config files in JSON format. Default is tokenizers.txt in the working directory.
	--compare: a boolean argument for comparing the segmentation difference between inference methods. Default is False. If enabled make sure the default segmentation is the first path in the tokenizers paths file (and that the vocabulary is shared by all tokenizers).
```
Example:
```    
python main.py \
        --tokenizers tokenizers.txt
```

## Citation

TBA

