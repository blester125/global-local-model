import logging
from itertools import chain
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from eight_mile.pytorch.layers import EmbeddingsStack, WithoutLength, WithDropout, ParallelConv
from baseline.pytorch import TensorDef, BaseLayer
from baseline.model import register_model
from baseline.pytorch.classify.model import EmbedPoolStackClassifier
from mead.utils import get_output_paths, create_metadata, save_to_bundle
from mead.exporters import Exporter, register_exporter
from mead.pytorch.exporters import ClassifyPytorchONNXExporter, FAKE_SENTENCE

logger = logging.getLogger('mead.layers')


class GlobalLocalEmbeddings(nn.Module):
    def __init__(self, global_emb, local_emb):
        super().__init__()
        self.global_emb = global_emb
        self.global_keys = set(self.global_emb.keys())
        self.local_emb = local_emb
        self.local_keys = set(self.local_emb.keys())

    def forward(self, inputs):
        return (
            self.global_emb({k: v for k, v in inputs.items() if k in self.global_keys}),
            self.local_emb({k: v for k, v in inputs.items() if k in self.local_keys}),
        )

    @property
    def output_dim(self):
        return tuple((self.global_emb.output_dim, self.local_emb.output_dim))

    def keys(self):
        return list(chain(self.global_keys, self.local_keys))

    def items(self):
        return list(chain(self.global_emb.items(), self.local_emb.items()))


class GlobalLocalPool(nn.Module):
    def __init__(self, global_pool: BaseLayer, local_pool: Optional[BaseLayer] = None):
        super().__init__()
        self.global_layer = global_pool
        self.local_layer = global_pool if local_pool is None else local_pool

    def forward(self, inputs: Tuple[TensorDef, TensorDef, TensorDef, TensorDef]) -> TensorDef:
        global_input, local_input, global_lengths, local_lengths = inputs
        global_view = self.global_layer((global_input, global_lengths))
        local_view = self.local_layer((local_input, local_lengths))
        return torch.cat([global_view, local_view], axis=-1)

    @property
    def output_dim(self):
        return self.global_layer.output_dim + self.local_layer.output_dim


class GlobalLocalClassifier(EmbedPoolStackClassifier):
    def init_embed(self, embeddings: Dict[str, BaseLayer], **kwargs) -> Tuple[BaseLayer, BaseLayer]:
        global_emb = {k: v for k, v in embeddings.items() if k.startswith('global')}
        local_emb = {k: v for k, v in embeddings.items() if k.startswith('local')}

        reduction = kwargs.get('embeddings_reduction', 'concat')
        embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))
        return GlobalLocalEmbeddings(
            EmbeddingsStack(global_emb, embeddings_dropout, reduction=reduction),
            EmbeddingsStack(local_emb, embeddings_dropout, reduction=reduction),
        )

    def forward(self, inputs):
        global_lengths = inputs.get("global_lengths")
        local_lengths = inputs.get("local_lengths")

        embedded = self.embeddings(inputs)
        embeddings = (*embedded, global_lengths, local_lengths)

        pooled = self.pool_model(embeddings)
        stacked = self.stack_model(pooled)
        return self.output_layer(stacked)


@register_model(task="classify", name="global-local-conv")
class GlobalLocalConvClassifier(GlobalLocalClassifier):
    def init_pool(self, input_dim: Tuple[int], **kwargs) -> BaseLayer:
        global_input_dim, local_input_dim = input_dim
        cmotsz = kwargs['cmotsz']
        filtsz = kwargs['filtsz']
        activation = kwargs.get('activation', 'relu')
        share = bool(kwargs.get("share_global_local", False))

        global_conv = WithoutLength(WithDropout(ParallelConv(global_input_dim, cmotsz, filtsz, activation), self.pdrop))
        if share:
            local_conv = global_conv
        else:
            local_conv = WithoutLength(
                WithDropout(
                    ParallelConv(
                        local_input_dim,
                        kwargs.get("local_cmotsz", cmotsz),
                        kwargs.get("local_filtsz", filtsz),
                        kwargs.get("local_activation", activation),
                    ),
                    kwargs.get("local_pdrop", self.pdrop),
                )
            )
        return GlobalLocalPool(global_conv, local_conv)


def create_json_data_dict(vocabs, vectorizers, transpose=False, min_=0, max_=50, default_size=100):
    data = {}
    example = {"utterance": FAKE_SENTENCE.split(), "span": FAKE_SENTENCE.split()}
    for k, v in vectorizers.items():
        data[k], feature_length = vectorizers[k].run(example, vocabs[k])
        data[k] = torch.LongTensor(data[k]).unsqueeze(0)
        data[f"{k}_lengths"] = torch.LongTensor([feature_length])

    if transpose:
        for k in vectorizers.keys():
            if len(data[k].shape) > 1:
                data[k] = data[k].transpose(0, 1)
    return data


@register_exporter(task="classify", name="json-vectorizer")
class JSONPytorchONNXExporter(ClassifyPytorchONNXExporter):
    def create_example_input(self, vocabs, vectorizers):
        return create_json_data_dict(vocabs, vectorizers, transpose=self.transpose, default_size=self.default_size)

    def create_model_inputs(self, model):
        return [k for k, _ in model.embeddings.items()] + ['lengths']


@register_exporter(task="classify", name="global-local")
class GlobalLocalJSONPytorchONNXExporter(JSONPytorchONNXExporter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.onnx_opset = 12

    def create_model_inputs(self, model):
        inputs = []
        for k, _ in model.embeddings.items():
            inputs.append(k)
            inputs.append(f"{k}_lengths")
        return inputs

    def create_dynamic_axes(self, model, vectorizers, inputs, outputs):
        # We need to have separate keys for the name of the dynamic `sequence` axis. If we don't ONNX things that the lengths
        # of these axis need to be same. It will try to reuse buffers and throw errors then the size of the full utterance
        # (global feature) is different that the size of the span (local feature)

        # Why do we default this to sequence and dynamic? In the classifier output it would be static and better named `classes`
        # This creation of dynamic arrays would be a great spot for an override hook
        dynamics = {'output': {1: 'sequence'}}
        for k, _ in model.embeddings.items():
            prefix = 'global' if k in model.embeddings.global_keys else 'local'
            if k == 'char':
                dynamics[k] = {1: f'{prefix}_sequence', 2: f'{prefix}_chars'}
            else:
                dynamics[k] = {1: f'{prefix}_sequence'}
        return dynamics
