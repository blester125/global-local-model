"""This is a classifier designed to label a span while still getting a look at the global context of the span."""

import logging
from operator import itemgetter
from collections import Counter
import numpy as np
from baseline.utils import listify, is_sequence
from baseline.services import ClassifierService
from baseline.reader import register_reader, CONLLSeqReader
from baseline.vectorizers import register_vectorizer, Token1DVectorizer, _token_iterator, Dict1DVectorizer


LOGGER = logging.getLogger("mead.layers")


class TokenMaskVectorizer(Token1DVectorizer):
    """Read in a list of 1s and 0s and convert them into ints representing a mask for the span."""

    def _next_element(self, tokens, _):
        for token in self.iterable(tokens):
            yield int(token)


@register_vectorizer("mask")
class DictTokenMaskVectorizer(TokenMaskVectorizer):
    """Read in a token level model from a conll file."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fields = listify(kwargs.get("fields", "mask"))
        self.delim = kwargs.get("token_delim", "@@")

    def iterable(self, tokens):
        return _token_iterator(self, tokens)


@register_vectorizer("classify-label")
class DictClassifyLabelVectorizer(Dict1DVectorizer):
    """Read the labels from a conll file but only return the labels that are inside the span.

    Note:
        This is designed with the assumption that there should only be a single label within a span
        (where the span is marked by the 1 locations).

        The label_vectorizer argument in the reader section of the config should be used to create this
        vectorizer inside the reader
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Read from the mask and the label column
        self.fields = listify(kwargs.get("fields", ["mask", "y"]))
        self.delim = kwargs.get("token_delim", "@@")

    def count(self, tokens):
        seen = 0
        counter = Counter()
        for tok in self.iterable(tokens):
            # Our tokens look like this {mask}@@{y} so we need to extract just the label for the label vocab
            # The Dict Vectorizers are designed to facilitate the combining of features for example
            # (word x POS tag) but we actually want to change our behavior for one feature given the
            # the value of the other one so we need to split these up.
            _, tok = tok.split(self.delim)
            counter[tok] += 1
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            # Our tokens look like this {mask}@@{y} so we need to extract just the label for the label vocab
            mask, label = atom.split(self.delim)
            # If our mask is a zero we are not in the span, only yield the label if we are in the span
            if mask != "0":
                yield vocab.get(label)

    def run(self, tokens, vocab):
        if self.mxlen < 0:
            self.mxlen = self.max_seen

        label = list(set(self._next_element(tokens, vocab)))
        if len(label) > 1:
            LOGGER.warning("Found multiple labels within a span, got: %s. Taking the first one (%s)", label, label[0])
        # Return the label for the span and the length which is always 1
        return label[0], 1


@register_reader(task="classify", name="conll")
class ClassifierCONLLSeqReader(CONLLSeqReader):
    """A small adapter to let us use the tagger task conll reader for the classify task."""

    def build_vocab(self, files, **kwargs):
        vs = super().build_vocab(files, **kwargs)
        # The classifier reader will return a list of labels as the second argument, convert the
        # label2index map into a list sorted by the index (the key with the value 0 will be first
        # then the one with a value of 1, etc)
        return vs, [k for k, _ in sorted(self.label2index.items(), key=itemgetter(1))]

    def load(self, filename, vocabs, batchsz, shuffle=False, sort_key=None):
        # The conll reader defaults to returning both the vocabs and the raw text which the classifier
        # doesn't expect so remove that here.
        data, _ = super().load(filename, vocabs, batchsz, shuffle, sort_key)
        return data


class SpanClassifierService(ClassifierService):
    def batch_input(self, tokens):
        """Out input should either be a List[Dict] representing 1 example of a List[List[Dict]] representing a batch."""
        if not is_sequence(tokens):
            raise ValueError(f"{self.__class__.__name__} requires List[List[Dict]] for List[Dict] as input")
        if isinstance(tokens[0], dict):
            tokens = [tokens]
        return tokens
