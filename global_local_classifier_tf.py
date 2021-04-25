from typing import Optional, Union, List, Dict, Tuple
import tensorflow as tf
from eight_mile.tf.layers import ParallelConv, WithDropout, WithoutLength
from baseline.model import register_model
from baseline.tf import TensorDef, BaseLayer
from baseline.tf.classify.model import EmbedPoolStackClassifier
from mead.exporters import register_exporter
from mead.tf.exporters import ClassifyTensorFlowExporter, create_assets, SignatureOutput


@tf.function(input_signature=[tf.TensorSpec(shape=(None, 2), dtype=tf.int64), tf.TensorSpec(shape=(), dtype=tf.int32)])
def to_dense_idx(indices, T):
    """Given indices in the form [row, col] ordered by row output where this index should go in a list of all possible indices.

    example on a 3, 3 output space

    The inputs are
        [0, 0],
        [0, 1],
        [1, 2],
        [2, 1],
        [2, 2],

    Adding these into a full list of indices would result in
        [0, 0],
        [0, 1],
        [0, 0],
        [1, 2],
        [0, 0],
        [0, 0],
        [2, 1],
        [2, 2],
        [0, 0],

    The index mapping (the output of this function)
        [0, 1, 3, 6, 7]

    """
    idx = tf.TensorArray(dtype=tf.int64, size=tf.shape(indices)[0], dynamic_size=False)
    prev = tf.cast(-1, tf.int64)
    j = tf.cast(0, tf.int64)
    T = tf.cast(T, tf.int64)
    for i in range(tf.shape(indices)[0]):
        if indices[i][0] == prev:
            j += 1
        else:
            prev = indices[i][0]
            j = tf.cast(0, tf.int64)
        idx = idx.write(i, prev * T + j)
    return idx.stack()


def to_dense_idx_graph(indices, T):
    i = tf.cast(0, tf.int32)
    j = tf.cast(0, tf.int64)
    T = tf.cast(T, tf.int64)
    prev = tf.cast(-1, tf.int64)
    idx = tf.TensorArray(dtype=tf.int64, size=tf.shape(indices)[0], dynamic_size=False)

    cond = lambda i, *args: tf.less(i, tf.shape(indices)[0])

    def body(i, p, j, idx):
        val = indices[i, 0]

        true = lambda: tf.add(j, 1)
        false = lambda: tf.cast(0, tf.int64)

        res = tf.cond(tf.equal(val, p), true, false)
        idx = idx.write(i, val * T + res)

        return tf.add(i, 1), val, res, idx

    r = tf.while_loop(cond, body, [i, prev, j, idx])
    return r[-1].stack()


def span_select(tensor, mask):
    """Select the values from tensor that have a 1 in mask and move the values so they are at the front."""
    B = tf.shape(mask)[0]
    T = tf.shape(mask)[1]

    # Get a tensor of [N, 2] that is the index of each 1 in the mask (there are N 1s in the mask)
    indices = tf.where(mask)
    # Figure out where each index would go in a list of all possible indices into the data. For example
    # if the first row had 1s at `[0, 2]` and `[0, 7]` these would be at index 0 and 1 because they are the
    # first two items even though they have a gap. The first item in second row would go at the first location
    # allowed for an item in that batch. For example if there are 8 columns then item `[1, 12]` would go at
    # index 8 which is the first place something in the second row could go

    # dense_indices = to_dense_idx(indices, T)
    dense_indices = to_dense_idx_graph(indices, T)
    # Scatter these indices out into an actual array that has `[0, 0]` for all the other indices
    indices = tf.scatter_nd(tf.reshape(dense_indices, (-1, 1)), indices, shape=(B * T, 2))

    # Get all the indices form the tensor, now each row will be front loaded with the values we want  and the
    # rest of the tensor will the data at index [0, 0]
    selected = tf.gather_nd(tensor, indices)
    dense = tf.reshape(selected, (B, T, -1))
    # Get the lengths of valid data from the mask
    dense_lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    dense_mask = tf.sequence_mask(dense_lengths, tf.shape(dense)[1])
    # Mask out the junk ([0, 0] values) from the result
    return tf.multiply(dense, tf.expand_dims(tf.cast(dense_mask, dense.dtype), -1)), dense_lengths


class GlobalLocalPool(tf.keras.layers.Layer):
    def __init__(self, global_pool: BaseLayer, local_pool: Optional[BaseLayer] = None, name: Optional[str] = None):
        """Create a pooled representation of a span.

        This model creates the pooled representation by a concatenation of the local pooled representation (the tokens
        in the span) and the global representation (all the tokens in the example).

        Note:
            If you don't pass a local_pool model the global_pool model will be reused.

        :param global_pool: The model that generates a pooled representation based on all the tokens `[B, T, H]` -> `[B, H]`
        :param local_pool: The model that generates a pooled representation based on the tokens in the span `[B, Ts, H]` -> `[B, H]`
        :param name: The name of the keras layer
        """
        super().__init__(name=name)
        self.global_layer = global_pool
        self.local_layer = global_pool if local_pool is None else local_pool

    def call(self, inputs: Tuple[TensorDef, TensorDef, TensorDef]) -> TensorDef:
        inputs, lengths, mask = inputs
        global_view = self.global_layer((inputs, lengths))
        spans, span_lengths = span_select(inputs, mask)
        local_view = self.local_layer((spans, span_lengths))
        return tf.concat([global_view, local_view], axis=-1)

    @property
    def output_dim(self):
        return self.global_layer.output_dim + self.local_layer.output_dim


class GlobalLocalParallelConv(GlobalLocalPool):
    def __init__(
        self,
        global_insz: Optional[int],
        global_outsz: Union[int, List[int]],
        global_filtsz: List[int],
        global_activation: str = "relu",
        global_pdrop: float = 0.5,
        global_name: Optional[str] = None,
        local_insz: Optional[int] = None,
        local_outsz: Optional[Union[int, List[int]]] = None,
        local_filtsz: Optional[List[int]] = None,
        local_activation: Optional[str] = None,
        local_pdrop: Optional[float] = None,
        local_name: Optional[str] = None,
        share: bool = False,
        name: Optional[str] = None,
        **kwargs,
    ):
        """Create a pooled representation of a span with ConvNets.

        :param global_insz: The input size of the global conv
        :param global_outsz: The output size of the global conv
        :param global_filtsz: The filter sizes of the global conv
        :param global_activation: The activation function used in the global conv
        :param global_pdrop: The dropout applied on top of the global conv
        :param global_name: The name of the keras layer used as the global conv
        :param local_insz: The input size of the local conv
        :param local_outsz: The output size of the local conv
        :param local_filtsz: The filter sizes of the local conv
        :param local_activation: The activation function used in the local conv
        :param local_pdrop: The dropout applied on top of the local conv
        :param local_name: The name of the keras layer used as the local conv
        :param share: Should we share the global and local pool model? If `True` all local parameters are ignored.
        :param name: The name of this keras layer
        """
        global_conv = WithoutLength(
            WithDropout(
                ParallelConv(global_insz, global_outsz, global_filtsz, global_activation, global_name), global_pdrop
            )
        )
        if share:
            local_conv = global_conv
        else:
            local_insz = global_insz if local_insz is None else local_insz
            local_outsz = global_outsz if local_outsz is None else local_outsz
            local_filtsz = global_filtsz if local_filtsz is None else local_filtsz
            local_activation = global_activation if local_activation is None else local_activation
            local_pdrop = global_pdrop if local_pdrop is None else local_pdrop
            local_name = global_name if local_name is None else local_name
            local_conv = WithoutLength(
                WithDropout(
                    ParallelConv(local_insz, local_outsz, local_filtsz, local_activation, local_name), local_pdrop
                )
            )
        super().__init__(global_conv, local_conv, name=name)


class GlobalLocalClassifier(EmbedPoolStackClassifier):
    def __init__(self, name=None):
        super().__init__(name=name)
        # These might added to the batch dict when we are using the conll tagger reader. When using tf.datasets these
        # fields are copied into the state dict. They are never used but we need remove them before serializing the
        # state
        self._unserializable.extend(["y_lengths", "ids"])

    def save_md(self, basename):
        super().save_md(basename)
        # Save out the dummy embeddings object. The mask comes into the model via the embeddings dict. In order to
        # rehydrate the model we need to have the mask in the embeddings. In order to do this we need to have an
        # Embedding object that can be recreated with a meta data dict.
        # Again this is a huge hack because we don't have the ability to divorce the inputs from the embeddings.
        self.mask_dummy_embedding.save_md(f"{basename}-mask-md.json")

    def init_embed(self, embeddings: Dict[str, TensorDef], **kwargs) -> BaseLayer:
        # Remove the mask from the embedding dict because we don't want to embed it. This is a hack to get around
        # the fact that baseline inputs are coupled to embedding objects. If we divorced these where something being
        # a feature just meant that it had a vectorizer associated with it and would show up in the batch dict (we would
        # need to generate placeholder information based on the vectorizer not on the embedding) then we wouldn't need
        # to do this hack.
        # Even if we had an embedding object that just passed the mask through untouched we couldn't use that because we
        # don't want the mask concatenated to the other embeddings.
        self.mask_name = kwargs.get("mask_name", "mask")
        # Save a reference to the dummy embedding object so we can save and rehydrate it later
        self.mask_dummy_embedding = embeddings.pop(self.mask_name, None)
        return super().init_embed(embeddings, **kwargs)

    def make_input(self, batch_dict, train=False):
        # Make the batched however the model normally does it.
        tf_batch = super().make_input(batch_dict, train)
        # Add the mask to the batch_dict, If this was the pyt version we would need to permute the mask
        # so it was ordered by lengths too.
        if not tf.executing_eagerly():
            tf_batch["{}:0".format(self.mask_name)] = batch_dict[self.mask_name]
        else:
            tf_batch[self.mask_name] = batch_dict[self.mask_name]
        return tf_batch

    def call(self, inputs: Dict[str, TensorDef]) -> TensorDef:
        """Run tensors through the model but pass the mask into the pooler too."""
        lengths = inputs.get("lengths")
        embedded = self.embeddings(inputs)
        embedded = (embedded, lengths, inputs[self.mask_name])
        pooled = self.pool_model(embedded)
        stacked = self.stack_model(pooled)
        return self.output_layer(stacked)


@register_model(task="classify", name="global-local-conv")
class GlobalLocalConvClassifier(GlobalLocalClassifier):
    def init_pool(self, input_dim: int, **kwargs) -> BaseLayer:
        """Create span representation model based on Conv Nets."""
        cmotsz = kwargs["cmotsz"]
        filtsz = kwargs["filtsz"]
        activation = kwargs.get("activation", "relu")
        name = kwargs.get("pool_name")
        return WithDropout(
            GlobalLocalParallelConv(
                input_dim,
                cmotsz,
                filtsz,
                global_activation=activation,
                global_pdrop=self.pdrop_value,
                global_name=kwargs.get("global_pool_name"),
                local_insz=kwargs.get("local_cmotsz"),
                local_outsz=kwargs.get("local_outsz"),
                local_filtsz=kwargs.get("local_filtsz"),
                local_activation=kwargs.get("local_activation"),
                local_pdrop=kwargs.get("local_pdrop"),
                local_name=kwargs.get("local_pool_name"),
                share=bool(kwargs.get("share_global_local", False)),
                name=name,
            ),
            self.pdrop_value,
        )


@register_exporter(task="classify", name="span-labeler")
class SpanLabelerTensorFlowExporter(ClassifyTensorFlowExporter):
    def _create_rpc_call(self, sess, basename, **kwargs):
        model, classes, values = self._create_model(sess, basename)

        predict_tensors = {}
        predict_tensors[model.lengths_key] = tf.compat.v1.saved_model.utils.build_tensor_info(model.lengths)

        for k, v in model.embeddings.items():
            try:
                predict_tensors[k] = tf.compat.v1.saved_model.utils.build_tensor_info(v.x)
            except:
                raise exception("unknown attribute in signature: {}".format(v))
        predict_tensors[model.mask_name] = tf.compat.v1.saved_model.utils.build_tensor_info(
            model.sess.graph.get_tensor_by_name(f"{model.mask_name}:0")
        )

        sig_input = predict_tensors
        sig_output = SignatureOutput(classes, values)
        sig_name = "predict_text"

        assets = create_assets(
            basename,
            sig_input,
            sig_output,
            sig_name,
            model.lengths_key,
            return_labels=self.return_labels,
            preproc=self.preproc_type(),
        )
        return sig_input, sig_output, sig_name, assets
