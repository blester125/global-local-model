"""This uses a Global/Local model to label spans in a conll file."""

import argparse
from typing import Optional
from baseline import read_conll, to_chunks, str2bool, create_progress_bar, import_user_module
from global_local_classifier import SpanClassifierService


def create_examples(
    sentence,
    surface_index: int,
    span_index: int,
    gold_index: int,
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
    golds = cols[gold_index]
    span_golds = ["O"] * len(golds)
    for chunk in to_chunks(tags, span_type, span_delim):
        _, *locs = chunk.split(span_delim)
        mask = ["0"] * len(tags)
        for loc in map(int, locs):
            mask[loc] = "1"
            span_golds[loc] = golds[loc]
        examples.append(list(zip(*[surfaces, mask])))
    example_dicts = []
    for example in examples:
        ex = []
        for token, mask in example:
            ex.append({surface_feature: token, mask_feature: mask})
        example_dicts.append(ex)
    return example_dicts, surfaces, span_golds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--conll", required=True)
    parser.add_argument("--surface_index", "--surface-index", type=int, required=True)
    parser.add_argument("--span_index", "--span-index", type=int, required=True)
    parser.add_argument("--gold_index", "--gold-index", type=int, required=True)
    parser.add_argument("--span_type", "--span-type", default="iobes", choices=("iobes", "bio", "iob"))
    parser.add_argument("--delim")
    parser.add_argument("--span_delim", "--span-delim", default="@")
    parser.add_argument("--surface_feature", "--surface-feature", default="text")
    parser.add_argument("--mask_feature", "--mask-feature", default="mask")
    parser.add_argument("--backend", default="tf")
    parser.add_argument('--remote', help='(optional) remote endpoint', type=str)
    parser.add_argument('--name', help='(optional) signature name', type=str)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batchsz", default=64, type=int)
    parser.add_argument('--prefer_eager', help="If running in TensorFlow, should we prefer eager model", type=str2bool)
    parser.add_argument("--output", default="output.conll")
    parser.add_argument("--modules", nargs="+", default=[])
    args = parser.parse_args()

    if args.backend == 'tf':
        from eight_mile.tf.layers import set_tf_eager_mode
        set_tf_eager_mode(args.prefer_eager)

    for mod in args.modules:
        import_user_module(mod)

    m = SpanClassifierService.load(args.model, backend=args.backend, device=args.device, remote=args.remote, name=args.name)

    examples = []
    surfaces = []
    golds = []
    for sentence in read_conll(args.conll, args.delim):
        ex, surf, gold = create_examples(
            sentence,
            args.surface_index,
            args.span_index,
            args.gold_index,
            args.span_type,
            args.delim,
            args.span_delim,
            args.surface_feature,
            args.mask_feature
        )
        if ex:
            examples.append(ex)
            surfaces.append(surf)
            golds.append(gold)

    outputs = []
    pg = create_progress_bar(len(examples))
    for batch, surf, gold in pg(zip(examples, surfaces, golds)):
        preds = ["O"] * len(surf)
        for label, example in zip(m.predict(batch), batch):
            indices = [i for i, m in enumerate(t[args.mask_feature] for t in example) if m == "1"]
            label = label[0][0]
            if label == "O":
                continue
            if len(indices) == 1:
                preds[indices[0]] = f"S-{label}"
                continue
            preds[indices[0]] = f"B-{label}"
            preds[indices[-1]] = f"E-{label}"
            for i in indices[1:-1]:
                preds[i] = f"I-{label}"
        outputs.append(
            list(zip(*[
                surf,
                gold,
                preds,
            ]))
        )

    with open(args.output, "w") as wf:
        wf.write("\n\n".join(["\n".join([" ".join(row) for row in sentence]) for sentence in outputs]))


if __name__ == "__main__":
    main()
