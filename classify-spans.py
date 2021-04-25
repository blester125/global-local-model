"""Given a conll file of spans, classify each span. This is more like classify-text and it is just for testing things out."""

import argparse
from typing import Optional
from baseline import read_conll, to_chunks, str2bool
from global_local_classifier import SpanClassifierService


def create_example(
    sentence,
    surface_index: int = 1,
    span_index: int = 4,
    span_type: str = "iobes",
    delim: Optional[str] = None,
    span_delim: str = "@",
    surface_feature: str = "text",
    mask_feature: str = "mask",
):
    examples = []
    cols = list(zip(*sentence))
    surfaces = cols[surface_index]
    tags = cols[span_index]
    for chunk in to_chunks(tags, span_type, span_delim):
        _, *locs = chunk.split(span_delim)
        mask = ["0"] * len(tags)
        for loc in map(int, locs):
            mask[loc] = "1"
        examples.append(list(zip(*[surfaces, mask])))
    example_dicts = []
    for example in examples:
        ex = []
        for token, mask in example:
            ex.append({surface_feature: token, mask_feature: mask})
        example_dicts.append(ex)
    return example_dicts



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--conll", required=True)
    parser.add_argument("--surface_index", "--surface-index", type=int, default=1)
    parser.add_argument("--span_index", "--span-index", type=int, default=4)
    parser.add_argument("--span_type", "--span-type", default="iobes", choices=("iobes", "bio", "iob"))
    parser.add_argument("--delim")
    parser.add_argument("--span_delim", "--span-delim", default="@")
    parser.add_argument("--surface_feature", "--surface-feature", default="text")
    parser.add_argument("--mask_feature", "--mask-feature", default="mask")
    parser.add_argument("--backend", default="tf")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batchsz", default=64, type=int)
    parser.add_argument('--prefer_eager', help="If running in TensorFlow, should we prefer eager model", type=str2bool)
    args = parser.parse_args()

    if args.backend == 'tf':
        from eight_mile.tf.layers import set_tf_eager_mode
        set_tf_eager_mode(args.prefer_eager)

    examples = []
    for sentence in read_conll(args.conll, args.delim):
        examples.extend(create_example(sentence, args.surface_index, args.span_index, args.span_type, args.delim, args.span_delim, args.surface_feature, args.mask_feature))

    m = SpanClassifierService.load(args.model, backend=args.backend, device=args.device)

    batched = [examples[i:i+args.batchsz] for i in range(0, len(examples), args.batchsz)]

    for batch in batched:
        for label, texts in zip(m.predict(batch), batch):
            span = []
            for token in texts:
                print(token["text"], end = " ")
                if token["mask"] == "1":
                    span.append(token["text"])
            print()
            print(label[0][0] + " -> " + " ".join(span) + "\n")


if __name__ == "__main__":
    main()
