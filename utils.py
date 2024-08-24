import json, os
from typing import List
from tokenizers import normalizers, pre_tokenizers


# map all bytes to valid utf-8 characters
# in the same way that the huggingface tokenizers byte level pretokenizer does
class HFEncoding:

    # translated from the rust code
    # see https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs
    @staticmethod
    def bytes_char():

        bs = []
        bs.extend(range(ord('!'), ord('~') + 1))
        bs.extend(range(0xA1, 0xAC + 1))
        bs.extend(range(0xAE, 0xFF + 1))
        cs = [b for b in bs]

        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1

        return {bytes([f]): chr(t) for f, t in zip(bs, cs)}

    def __init__(self):
        # map any byte to the corresponding character
        self.byte_map = HFEncoding.bytes_char()
        # the inverse character to byte mapping
        self.inv_byte_map = {v: k for k, v in self.byte_map.items()}

    # convert an encoded string of our mapped characters back to the original bytes
    def tobytes(self, s: str) -> bytes:
        return b"".join([self.inv_byte_map[c] for c in s])

    # convert a byte string into an encoded string of valid characters
    def toencoded(self, byte_str: bytes) -> str:
        return "".join([self.byte_map[bytes([c])] for c in byte_str])


# read our hex formatted vocab file
# return a list of bytes objects
# input file has one vocab word per line each hex encoded
def load_tokenizer(config_filepath):
    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f'Missing vocab file: {config_filepath}')

    with open(config_filepath, 'r') as config_file:
        tokenizer_config = json.load(config_file)
    return tokenizer_config


def corpus_to_list(file_path: str, encoding: str = "utf-8") -> List[str]:
    corpus = []
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            for line in file:
                corpus.append(line)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    return corpus


def get_hf_normalizer(normalizer_config):
    match normalizer_config['type']:
        case 'BertNormalizer':
            return normalizers.BertNormalizer(**filter_config(normalizer_config))
        case 'Lowercase':
            return normalizers.Lowercase()
        case 'NFC':
            return normalizers.NFC()
        case 'NFD':
            return normalizers.NFD()
        case 'NFKC':
            return normalizers.NFKC()
        case 'NFKD':
            return normalizers.NFKD()
        case 'Nmt':
            return normalizers.Nmt()
        case 'Precompiled':
            precompiled_charsmap = bytes(normalizer_config['precompiled_charsmap'], 'utf-32')
            return normalizers.Precompiled(precompiled_charsmap)
        case 'Replace':
            return normalizers.Replace(**filter_config(normalizer_config))
        case 'Strip':
            return normalizers.Strip(**filter_config(normalizer_config))
        case 'StripAccents':
            return normalizers.StripAccents()


def get_hf_pretokenizer(pretokenizer_config):
    match pretokenizer_config['type']:
        case 'BertPreTokenizer':
            return pre_tokenizers.BertPreTokenizer()
        case 'ByteLevel':
            return pre_tokenizers.ByteLevel(**filter_config(pretokenizer_config))
        case 'CharDelimiterSplit':
            return pre_tokenizers.CharDelimiterSplit()
        case 'Digits':
            return pre_tokenizers.Digits(**filter_config(pretokenizer_config))

        case 'Metaspace':
            return pre_tokenizers.Metaspace(**filter_config(pretokenizer_config))
        case 'Punctuation':
            return pre_tokenizers.Punctuation(**filter_config(pretokenizer_config))
        case 'Split':
            pretokenizer_config['pattern'] = pretokenizer_config['pattern']['Regex']
            pretokenizer_config['behavior'] = pretokenizer_config['behavior'].lower()
            return pre_tokenizers.Split(**filter_config(pretokenizer_config))
        case 'UnicodeScripts':
            return pre_tokenizers.UnicodeScripts()
        case 'Whitespace':
            return pre_tokenizers.Whitespace()
        case 'WhitespaceSplit':
            return pre_tokenizers.WhitespaceSplit()


def filter_config(config: dict):
    return {key: value for key, value in config.items() if key != 'type' and value is not None}

