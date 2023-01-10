import numpy as np
from seqio import Vocabulary

import ast
import regex as re
from typing import Optional, Sequence, Iterable, Union
import tensorflow as tf


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class GPT2Vocabulary(Vocabulary):
    """Abstract class for all vocabularies.

    Subclasses must implement methods for converting between strings and tokens
    both in pure python (`_encode`/`_decode`) and in TensorFlow
    (`_encode_tf`/`_decode_tf`).

    Subclasses are responsible for reserving PAD_ID=0 as well as optionally
    reserving EOS_ID and UNK_ID

    `_base_vocab_size` should account for PAD, EOS, and UNK but not `extra_ids`.
    """

    def __init__(self, extra_ids: int = 0):
        """Vocabulary constructor. GPT2 does not have any extra tokens. """
        self._extra_ids = 0
        vocab_file = "/home/helen/data/vocab.json"  # TODO(helen) get this from gcs lol
        vocab_bpe = "/home/helen/data/vocab.bpe"
        with open(vocab_file, 'r') as f:
            self.encoder = ast.literal_eval(f.read())
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = 'replace'  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with open(vocab_bpe, 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def eos_id(self) -> Optional[int]:
        raise 50256

    # @property
    # def pad_id(self) -> int:
    #   return PAD_ID
    #
    # @abc.abstractproperty
    # def unk_id(self) -> Optional[int]:
    #   raise NotImplementedError("need to implement unk_id")

    # @property
    # def extra_ids(self) -> int:
    #   return self._extra_ids

    @property
    def vocab_size(self) -> int:
        """Vocabulary size, including extra ids."""
        return self._base_vocab_size + self.extra_ids

    def _base_vocab_size(self) -> int:
        """Vocabulary size, excluding extra ids but including PAD/EOS/UNK."""
        # TODO(fjord): add a check that pad_id and unk_id (if present)
        #   are less than _base_vocab_size.
        return 50257

    def _encode(self, s: str) -> Sequence[int]:
        text = s
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def encode(self, s: Union[Sequence[int], str]) -> Sequence[int]:
        """Tokenizes string to an int sequence, without adding EOS."""
        return self._encode(s)

    def _decode(self, ids):
        tokens = ids
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

    def decode(self, ids: Iterable[int]):
        """Detokenizes int32 iterable to a string, up through first EOS."""
        clean_ids = list(ids)

        if self.unk_id is not None:
            vocab_size = self._base_vocab_size
            clean_ids = [
                self.unk_id if i >= vocab_size else i
                for i in clean_ids
            ]

        if self.eos_id is not None and self.eos_id in clean_ids:
            clean_ids = clean_ids[:clean_ids.index(self.eos_id) + 1]

        return self._decode(clean_ids)

    def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(self._encode(s), dtype=tf.int32)

    def encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        """Tokenizes string Scalar to an int32 Tensor, without adding EOS."""
        return self._encode_tf(s)

    def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
        if not isinstance(ids, (list, tf.Tensor, np.ndarray)):  # a single token
            ids = np.expand_dims(np.int32(ids), axis=0)
        elif isinstance(ids, tf.Tensor):
            ids = ids.numpy()

        tokens = []
        for token in ids:
            tokens.append(self.decoder[token])
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        return text

    def decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
        """Detokenizes int32 batched Tensor through first EOS."""
        clean_ids = ids

        if self.unk_id is not None:
            clean_ids = tf.where(
                tf.less(clean_ids, self._base_vocab_size), clean_ids, self.unk_id)

        if self.eos_id is not None:
            # Replace everything after the first eos_id with pad_id.
            after_eos = tf.cumsum(
                tf.cast(tf.equal(clean_ids, self.eos_id), tf.int32),
                exclusive=True, axis=-1)
            clean_ids = tf.where(tf.cast(after_eos, tf.bool), self.pad_id, clean_ids)

        return self._decode_tf(clean_ids)
