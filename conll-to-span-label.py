"""Convert a normal conll file with spans into a conll file for classifying a specific span.

Note:
    This might have been easier to just pass in the whole utterance and the span? We would
    lose out on the shared embedding if we did that though.

Given a sentence like

Bill  B-PER
Ford  E-PER
is    O
cool  O
as    O
is    O
Jason S-PER

Convert each span into an example for classification

Bill  1 PER
Ford  1 PER
is    0 O
cool  0 O
as    0 O
is    0 O
Jason 0 O

Bill  0 O
Ford  0 O
is    0 O
cool  0 O
as    0 O
is    0 O
Jason 1 PER

The resulting columns are the surface terms, a binary mask representing if that token is
part of the span we are classifying (we are using a mask to allow for gaps in the IOBES
representation of the span) and the label for that span.
"""

import argparse
from typing import Optional, List
from collections import Counter
from baseline import read_conll, to_chunks


def extract_label(label: str) -> str:
    """Extract the label from a conll field that might have
           {token-function}-{type}"
    """
    if "-" not in label:
        return label
    return label.split("-", maxsplit=1)[1]


def create_masked_examples(
    sentence: List[List[str]],
    surface_index: int,
    span_index: int,
    label_index: int,
    span_type: str = "iobes",
    delim: Optional[str] = None,
) -> List[List[List[str]]]:
    """Convert a conll sentence into a list of sentences. Each span will generate a new sample.

    :param sentence: The conll sentence
    :param surface_index: The column index of the text features
    :param span_index: The column index of the span tags. This is decoded into chunks and each chunk
        will generate a new example with that chunk as the parts with the valid mask.
    :param label_index: the column to pull the label out of. basically we get all the token in the
        mask and take the most common label.
    :param span_type: How the chunks in span_index are encoded
    :param delim: A special character used to separate chunk type and locations. This character
        must not be in any of the type names

    :returns: The list of examples generated from the sentence
    """
    examples = []
    cols = list(zip(*sentence))
    surfaces = cols[surface_index]
    features = cols[span_index]
    label_col = cols[label_index]
    for chunk in to_chunks(features, span_type, delim):
        _, *locs = chunk.split(delim)
        locs = list(map(int, locs))
        labels = ["O"] * len(features)
        mask = ["0"] * len(features)
        label = Counter(extract_label(label_col[l]) for l in locs).most_common()[0][0]
        for loc in locs:
            mask[loc] = "1"
            labels[loc] = label
        examples.append(list(zip(*[surfaces, mask, labels])))
    return examples


def create_dataset(
    file_name: str,
    surface_index: int,
    span_index: int,
    label_index: int,
    span_type: str = "iobes",
    delim: Optional[str] = None,
    span_delim: str = "@",
):
    """Convert each conll sentence into a list of sentence. Each span will generate a new sample.

    :param file_name: The conll file to read from
    :param surface_index: The column index of the text features
    :param span_index: The column index of the span tags. This is decoded into chunks and each chunk
        will generate a new example with that chunk as the parts with the valid mask.
    :param label_index: the column to pull the label out of. basically we get all the token in the
        mask and take the most common label.
    :param span_type: How the chunks in span_index are encoded
    :param delim: A special character to split conll columns on
    :param span_delim: A special character used to separate chunk type and locations. This character
        must not be in any of the type names

    :returns: The list of examples generated from each sentence
    """
    examples = []
    for sentence in read_conll(file_name, delim):
        examples.extend(create_masked_examples(sentence, surface_index, span_index, label_index, span_type, span_delim))
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Convert a conll file of spans into a dataset for classifying particular spans."
    )
    parser.add_argument("input", help="The input conll file")
    parser.add_argument("output", help="The name of the output file")
    parser.add_argument(
        "--surface_index", "--surface-index", type=int, help="The index of the surface terms", required=True
    )
    parser.add_argument(
        "--span_index", "--span-index", type=int, help="The column to parse span spans from", required=True
    )
    parser.add_argument(
        "--label_index", "--label-index", type=int, help="The column to extract the labels from", required=True
    )
    parser.add_argument(
        "--span_type",
        "--span-type",
        default="iobes",
        choices=("iobes", "bio", "iob"),
        help="The encoding scheme used by spans in `--span-index`",
    )
    parser.add_argument("--delim", help="A specific separator of conll columns.")
    parser.add_argument(
        "--span_delim",
        "--span-delim",
        default="@",
        help="A character used to separate the chunk type from the indices, needs to be a symbol that doesn't appear in a type.",
    )
    args = parser.parse_args()

    output_delim = " " if args.delim is None else args.delim
    new_data = create_dataset(
        args.input, args.surface_index, args.span_index, args.label_index, args.span_type, args.delim, args.span_delim
    )

    with open(args.output, "w") as wf:
        wf.write(
            "\n\n".join(["\n".join([output_delim.join(r for r in row) for row in sentence]) for sentence in new_data])
            + "\n\n"
        )


if __name__ == "__main__":
    main()
