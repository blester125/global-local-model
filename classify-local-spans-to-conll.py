"""This uses a normal classifier (only looks at the span) to create a conll file where each span is labeled."""

import argparse
from typing import Optional
from baseline import read_conll, to_chunks, str2bool, create_progress_bar
from baseline.services import ClassifierService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--conll", required=True)
    parser.add_argument("--surface_index", "--surface-index", type=int, default=1)
    parser.add_argument("--span_index", "--span-index", type=int, default=4)
    parser.add_argument("--gold_index", "--gold-index", type=int, default=6)
    parser.add_argument("--span_type", "--span-type", default="iobes", choices=("iobes", "bio", "iob"))
    parser.add_argument("--delim")
    parser.add_argument("--span_delim", "--span-delim", default="@")
    parser.add_argument("--backend", default="tf")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batchsz", default=64, type=int)
    parser.add_argument("--output", default="output.conll")
    parser.add_argument('--prefer_eager', help="If running in TensorFlow, should we prefer eager model", type=str2bool)
    args = parser.parse_args()

    if args.backend == 'tf':
        from eight_mile.tf.layers import set_tf_eager_mode
        set_tf_eager_mode(args.prefer_eager)

    m = ClassifierService.load(args.model, backend=args.backend, device=args.device)

    outputs = []
    examples = list(read_conll(args.conll))
    pg = create_progress_bar(len(examples))
    for sentence in pg(examples):
        preds = ["O"] * len(sentence)
        golds = ["O"] * len(sentence)
        batch = []
        indices = []
        cols = list(zip(*sentence))
        surfaces = cols[args.surface_index]
        tags = cols[args.span_index]
        span_golds = cols[args.gold_index]
        for chunk in to_chunks(tags, args.span_type, args.span_delim):
            _, *locs = chunk.split(args.span_delim)
            locs = list(map(int, locs))
            batch.append([surfaces[l] for l in locs])
            indices.append(locs)
        if not batch:
            continue
        for i, label in enumerate(m.predict(batch)):
            label = label[0][0]
            if label == "O":
                continue
            locs = indices[i]
            for l in locs:
                golds[l] = span_golds[l]
            if len(locs) == 1:
                preds[locs[0]] = f"S-{label}"
                continue
            preds[locs[0]] = f"B-{label}"
            preds[locs[-1]] = f"E-{label}"
            for i in locs[1:-1]:
                preds[i] = f"I-{label}"
        outputs.append(
            list(zip(*[
                surfaces,
                golds,
                preds,
            ]))
        )

    with open(args.output, "w") as wf:
        wf.write("\n\n".join(["\n".join([" ".join(row) for row in sentence]) for sentence in outputs]))


if __name__ == "__main__":
    main()
