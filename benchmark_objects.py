from utils import get_hf_normalizer, get_hf_pretokenizer, load_tokenizer
from tokenizers import models, normalizers, pre_tokenizers
import copy


class BenchmarkTokenizer:

    def __init__(self, config_filepath):
        self.config = load_tokenizer(config_filepath)
        self.normalizer = BenchmarkNormalizer(self.config['normalizer'])
        self.pre_tokenizer = BenchmarkPreTokenizer(self.config['pre_tokenizer'])
        self.model = BenchmarkModel(self.config['model'])

        model_config = self.config['model']
        self.type = model_config['type']
        if isinstance(model_config['vocab'], dict):
            self.vocab: dict[bytes:int] = model_config['vocab']
        elif isinstance(model_config['vocab'], list):
            # this is the case of a unigram based vocab where each inner list is (token,likelihood)
            self.vocab = {ls[0]: idx for idx, ls in enumerate(model_config['vocab'])}
        self.inv_vocab: dict[int:bytes] = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def tokenize(self, text):
        normalized_text = self.normalizer.normalize_str(text)
        pre_tokenized_text = self.pre_tokenizer.pre_tokenize_str(normalized_text)
        tokens = []
        for word, offset in pre_tokenized_text:
            tokens.extend(map(lambda tok: tok.value, self.model.tokenize(word)))
        if self.get_type() == "WP_equal_like":
            tokens = list(map(lambda tok: "##" + tok, tokens))
            tokens[0] = tokens[0][2:]
        return tokens

    def get_vocab(self):
        return self.vocab

    def is_byte_level(self):
        return self.pre_tokenizer.byte_level

    def get_type(self):
        return self.model.type


class BenchmarkNormalizer:

    def __init__(self, config_normalizer):
        self.backend_normalizer = None
        if config_normalizer:
            if config_normalizer['type'] == "Sequence":
                hf_normalizers = [get_hf_normalizer(config) for config in config_normalizer['normalizers']]
                self.backend_normalizer = normalizers.Sequence(hf_normalizers)
            else:
                self.backend_normalizer = get_hf_normalizer(config_normalizer)

    def normalize_str(self, sequence):
        """
        Normalize the given string

        This method provides a way to visualize the effect of a Normalizer,
         but it does not keep track of the alignment information.
         If you need to get/convert offsets, you can use normalize()
        :param normalized:
        :return:
        """
        if self.backend_normalizer:
            return self.backend_normalizer.normalize_str(sequence)
        else:
            return sequence


class BenchmarkPreTokenizer:

    def __init__(self, pretokenizer_config):
        self.byte_level = False
        if pretokenizer_config:
            if pretokenizer_config['type'] == "Sequence":
                hf_pretokenizers = [get_hf_pretokenizer(config) for config in pretokenizer_config['pretokenizers']]
                for config in pretokenizer_config['pretokenizers']:
                    self.byte_level = self.byte_level or config['type'] == 'ByteLevel'
                self.backend_pretokenizer = pre_tokenizers.Sequence(hf_pretokenizers)
            else:
                self.backend_pretokenizer = get_hf_pretokenizer(pretokenizer_config)
                self.byte_level = pretokenizer_config['type'] == "ByteLevel"
        else:
            self.backend_pretokenizer = None

    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a PreTokenizer,
        but it does not keep track of the alignment, nor does it provide all the capabilities
        of the PreTokenizedString. If you need some of these, you can use pre_tokenize()
        :param sequence:
        :return:
        """
        if self.backend_pretokenizer:
            return self.backend_pretokenizer.pre_tokenize_str(sequence)
        else:
            return sequence.split()


class BenchmarkModel:

    def __init__(self, model_config):
        self.type = model_config['type']
        model_config = copy.deepcopy(model_config)
        match model_config.pop('type'):
            case 'BPE':
                model_config['merges'] = tuple(tuple(s.split(" ")) for s in model_config['merges'])
                if not model_config['continuing_subword_prefix']:
                    model_config['continuing_subword_prefix'] = ""
                if not model_config['end_of_word_suffix']:
                    model_config['end_of_word_suffix'] = ""
                self.backend_model = models.BPE(**model_config)
            case 'WordPiece':
                self.backend_model = models.WordPiece(**model_config)
            case 'BPE_dropout':
                model_config['merges'] = tuple(tuple(s.split(" ")) for s in model_config['merges'])
                if not model_config['continuing_subword_prefix']:
                    model_config['continuing_subword_prefix'] = ""
                if not model_config['end_of_word_suffix']:
                    model_config['end_of_word_suffix'] = ""
                self.backend_model = models.BPE(**model_config)
            case 'Unigram':
                model_config['vocab'] = tuple(tuple(ls) for ls in model_config['vocab'])
                self.backend_model = models.Unigram(**model_config)
            case 'WordLevel':
                self.backend_model = models.WordLevel(**model_config)
            case 'Sage':
                self.backend_model = models.WordPiece(**model_config)
            case 'Greedy_Unigram':
                self.backend_model = models.WordPiece(**model_config)
            case 'Greedy_BPE':
                self.backend_model = models.WordPiece(**model_config)
            case 'SaGe_as_Unigram':
                model_config['vocab'] = tuple(tuple(ls) for ls in model_config['vocab'])
                self.backend_model = models.Unigram(**model_config)
            case 'Unigram_equal_like':
                model_config['vocab'] = tuple(tuple(ls) for ls in model_config['vocab'])
                self.backend_model = models.Unigram(**model_config)
            case 'BPE_equal_like':
                model_config['vocab'] = tuple(tuple(ls) for ls in model_config['vocab'])
                self.backend_model = models.Unigram(**model_config)
            case 'SaGe_equal_like':
                model_config['vocab'] = tuple(tuple(ls) for ls in model_config['vocab'])
                self.backend_model = models.Unigram(**model_config)
            case 'WP_equal_like':
                model_config['vocab'] = tuple(tuple(ls) for ls in model_config['vocab'])
                self.backend_model = models.Unigram(**model_config)
            case 'flota':
                # unigram
                if isinstance(model_config['vocab'], list):
                    vocab = tuple(tuple(ls) for ls in model_config['vocab'])
                    model_config["vocab"] = {tok[0]: i for i, tok in enumerate(vocab)}
                self.backend_model = FlotaTokenizer(model_config["vocab"])
            case 'WP_flota':
                # unigram
                if isinstance(model_config['vocab'], list):
                    vocab = tuple(tuple(ls) for ls in model_config['vocab'])
                    model_config["vocab"] = {tok[0]: i for i, tok in enumerate(vocab)}
                self.backend_model = FlotaTokenizer(model_config["vocab"], special="##")
            case 'longest_suffix':
                if isinstance(model_config['vocab'], list):
                    vocab = tuple(tuple(ls) for ls in model_config['vocab'])
                    model_config["vocab"] = {tok[0]: i for i, tok in enumerate(vocab)}
                self.backend_model = LongestSuffix(model_config["vocab"])
            case 'WP_longest_suffix':
                if isinstance(model_config['vocab'], list):
                    vocab = tuple(tuple(ls) for ls in model_config['vocab'])
                    model_config["vocab"] = {tok[0]: i for i, tok in enumerate(vocab)}
                self.backend_model = LongestSuffix(model_config["vocab"], special="##")

    def tokenize(self, sequence):
        return self.backend_model.tokenize(sequence)


class FlotaTokenizer:
    def __init__(self, vocab, special="Ġ"):
        self.vocab = vocab
        self.special = special

    def max_subword_split(self, w):
        for l in range(len(w), 0, -1):
            for i in range(0, len(w) - l + 1):
                if w[i] == "\u2581":
                    continue
                subword = w[i:i + l]
                if self.special == "Ġ":
                    if subword in self.vocab:
                        return subword, w[:i] + l * "\u2581" + w[i + l:], i
                else:
                    if i == 0:
                        if subword in self.vocab:
                            return subword, w[:i] + l * "\u2581" + w[i + l:], i
                    else:
                        if (self.special + subword) in self.vocab:
                            return self.special + subword, w[:i] + l * "\u2581" + w[i + l:], i
        return None, None, None

    def get_flota_dict(self, w):
        max_subword, rest, i = self.max_subword_split(w)
        if max_subword is None:
            return dict()
        if rest == len(rest) * "\u2581":
            flota_dict = {i: max_subword}
            return flota_dict
        flota_dict = self.get_flota_dict(rest)
        flota_dict[i] = max_subword
        return flota_dict

    def tokenize(self, w):
        flota_dict = self.get_flota_dict(w)
        return [Token(subword) for i, subword in sorted(flota_dict.items())]


class LongestSuffix:
    def __init__(self, vocab, special="Ġ"):
        self.vocab = vocab
        self.special = special

    def tokenize(self, w):
        tokens = []
        i = 0
        while w and i < len(w):
            if self.special == "Ġ":
                if w[i:] in self.vocab:
                    tokens.insert(0, w[i:])
                    w = w[:i]
                    i = 0
                else:
                    i += 1
            else:
                if i == 0:
                    if w[i:] in self.vocab:
                        tokens.insert(0, w[i:])
                        w = w[:i]
                        i = 0
                    else:
                        i += 1
                else:
                    if self.special + w[i:] in self.vocab:
                        tokens.insert(0, self.special + w[i:])
                        w = w[:i]
                        i = 0
                    else:
                        i += 1
        return [Token(token) for token in tokens]


class Token:
    def __init__(self, subword):
        self.value = subword
