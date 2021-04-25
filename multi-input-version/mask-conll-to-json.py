import os
import json
import logging
import argparse
from typing import List, Dict, Union, Optional
from baseline.utils import read_conll, remove_conll_extensions


logging.basicConfig(
    format="[Mask Conll → Json] %(message)s",
    level=getattr(logging, os.getenv("MASK_CONLL_2_JSON_LOG_LEVEL", "WARNING").upper(), logging.WARN),
)
LOGGER = logging.getLogger()


def convert_to_json(
    file_name: str, surf_idx: int, mask_idx: int, label_idx: int, delim: Optional[str] = None
) -> List[Dict[str, Union[str, List[str]]]]:
    examples = []
    for sentence in read_conll(file_name, delim):
        cols = list(zip(*sentence))
        surfs = cols[surf_idx]
        mask = cols[mask_idx]
        labels = cols[label_idx]

        span = [surfs[i] for i, m in enumerate(mask) if m == "1"]
        label = set(labels[i] for i, m in enumerate(mask) if m == "1")

        if len(label) != 1:
            LOGGER.warning("Found multiple labels in a span\n%s\n%s\n%s", surfs, mask, labels)
        label = list(label)[0]

        examples.append({"utterance": surfs, "span": span, "label": label})
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Convert the span label conll file (with the masking) into a json based representation"
    )
    parser.add_argument("conll")
    parser.add_argument("--output")
    parser.add_argument(
        "--surface_index", "--surface-index", type=int, default=0, help="The location of the text of an utterance"
    )
    parser.add_argument(
        "--mask_index",
        "--mask-index",
        type=int,
        default=1,
        help="The column index of the mask denoting spans with a `1`",
    )
    parser.add_argument(
        "--label_index", "--label-index", type=int, default=-1, help="The column index of the label for a span"
    )
    parser.add_argument("--jsonl", action="store_true", help="Should we write each Json example as a line in a file?")
    parser.add_argument(
        "--indent",
        type=int,
        help="The indent to use when storing as a single .json file. Ignored when `--jsonl` is active",
    )
    parser.add_argument("--delim")
    args = parser.parse_args()

    examples = convert_to_json(args.conll, args.surface_index, args.mask_index, args.label_index, args.delim)

    if args.output is None:
        ext = ".jsonl" if args.jsonl else ".json"
        args.output = f"{remove_conll_extensions(args.conll)}{ext}"

    if args.jsonl:
        with open(args.output, "w") as wf:
            for ex in examples:
                wf.write(json.dumps(ex) + "\n")
    else:
        with open(args.output, "w") as wf:
            json.dump(examples, wf, indent=args.indent)


if __name__ == "__main__":
    main()
