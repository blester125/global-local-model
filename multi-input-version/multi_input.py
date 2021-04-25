import os
import json
from collections import Counter, defaultdict
from operator import itemgetter, attrgetter
import numpy as np
import baseline
from baseline.utils import listify, unzip_files, find_model_basename, load_vocabs, load_vectorizers, read_json
from baseline.reader import register_reader
from baseline.services import ONNXClassifierService
from baseline.vectorizers import create_vectorizer, register_vectorizer, register_vectorizer, Token1DVectorizer


@register_vectorizer(name="json-token1d")
class JSONToken1DVectorizer(Token1DVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get("field", "text")

    def iterable(self, example):
        for tok in example[self.field]:
            yield self.transform_fn(tok)


@register_vectorizer(name="json-label")
class JSONLabelVectorizer(Token1DVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get("field", "label")

    def iterable(self, example):
        for tok in listify(example[self.field]):
            yield self.transform_fn(tok)


@register_reader(task="classify", name="json")
class JSONReader:
    def __init__(self, vectorizers, trim=True, truncate=False, mxlen=-1, **kwargs):
        super().__init__()
        self.vectorizers = vectorizers
        self.trim = trim
        self.truncate = truncate
        label_vectorizer_spec = kwargs.get('label_vectorizer', None)
        if label_vectorizer_spec:
            self.label_vectorizer = create_vectorizer(**label_vectorizer_spec)
        else:
            self.label_vectorizer = JSONLabelVectorizer(field='y', mxlen=mxlen)
        self.labels = Counter()

    def load_examples(self, files):
        examples = []
        for file_name in files:
            with open(file_name) as f:
                examples.extend(json.load(f))
        return examples

    def build_vocab(self, files, **kwargs):
        examples = self.load_examples(files)
        vocab = {k: Counter() for k in self.vectorizers.keys()}
        for example in examples:
            for k, vectorizer in self.vectorizers.items():
                vocab[k].update(vectorizer.count(example))
            self.labels.update(self.label_vectorizer.count(example))
        self.labels = {k: i for i, k in enumerate(self.labels.keys())}
        return vocab, [k for k, _ in sorted(self.labels.items(), key=itemgetter(1))]

    def load(self, file_name, vocabs, batchsz, shuffle=False, sort_key=None):
        if sort_key is not None and not sort_key.endswith('_lengths'):
            sort_key = f"{sort_key}_lengths"

        loaded = []
        examples = self.load_examples(listify(file_name))

        for example in examples:
            example_dict = {}
            for k, vectorizer in self.vectorizers.items():
                example_dict[k], lengths = vectorizer.run(example, vocabs[k])
                if lengths is not None:
                    example_dict[f'{k}_lengths'] = lengths
            example_dict['y'], _ = self.label_vectorizer.run(example, self.labels)
            example_dict['y'] = example_dict['y'].item()
            loaded.append(example_dict)
        return baseline.data.ExampleDataFeed(
            baseline.data.DictExamples(loaded, do_shuffle=shuffle, sort_key=sort_key),
            batchsz=batchsz,
            shuffle=shuffle,
            trim=self.trim,
            truncate=self.truncate,
        )


@register_reader(task="classify", name="jsonl")
class JSONlReader(JSONReader):
    def load_examples(self, files):
        for file_name in files:
            with open(file_name) as f:
                for line in f:
                    line = line.rstrip("\n")
                    yield json.loads(line)


class MultiInputONNXClassifierService(ONNXClassifierService):
    def vectorize(self, tokens_batch):
        """Turn the input into that batch dict for prediction.

        :param tokens_batch: `List[List[str]]`: The input text batch.

        :returns: dict[str] -> np.ndarray: The vectorized batch.
        """
        examples = defaultdict(list)
        if self.lengths_key is None:
            self.lengths_key = list(self.vectorizers.keys())[0]

        for i, tokens in enumerate(tokens_batch):
            for k, vectorizer in self.vectorizers.items():
                vec, length = vectorizer.run(tokens, self.vocabs[k])
                examples[k].append(vec)
                examples[f"{k}_lengths"].append(length)
        for k in self.vectorizers.keys():
            examples[k] = np.stack(examples[k])
            examples[f"{k}_lengths"] = np.stack(examples[f"{k}_lengths"])
        return examples
