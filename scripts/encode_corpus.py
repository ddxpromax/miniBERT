"""Encode a cleaned JSONL corpus using an existing vocabulary."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from minibert.data.corpus import iter_corpus_texts
from minibert.data.encoding import iter_encoded_documents
from minibert.data.token_storage import TokenizedCorpus, write_tokenized_corpus
from minibert.tokenizer.bert import BertTokenizer
from minibert.tokenizer.vocabulary import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")

    args = parser.parse_args()

    if args.max_documents is not None and args.max_documents < 1:
        parser.error("--max-documents must be at least 1")
    
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    
    if not args.vocab.is_file():
        parser.error(f"vocabulary file does not exist: {args.vocab}")
    
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")
    
    return args

def main() -> None:
    args = parse_args()
    started_at = perf_counter()

    vocabulary = Vocabulary.from_file(args.vocab)
    tokenizer = BertTokenizer(vocabulary, do_lower_case=True)

    texts = iter_corpus_texts(args.input)

    if args.max_documents is not None:
        texts = islice(texts, args.max_documents)
    
    with tqdm(
        texts,
        desc="Encoding",
        unit="doc",
        disable=args.no_progress,
    ) as progress:
        documents = iter_encoded_documents(progress, tokenizer)

        write_tokenized_corpus(
            documents,
            args.output,
            vocabulary,
        )
    
    corpus = TokenizedCorpus(args.output)

    print(f"Output: {args.output}")
    print(f"Document: {len(corpus):,}")
    print(f"Tokens: {corpus.num_tokens:,}")
    print(f"Elapsed: {perf_counter() - started_at:.1f} seconds")

    if len(corpus) > 0:
        first_document = corpus[0]
        preview_ids = first_document[:20].tolist()

        print(f"First document length: {len(first_document):,}")
        print(f"First 20 IDs: {preview_ids}")
        print(
            "First 20 tokens:",
            tokenizer.convert_ids_to_tokens(preview_ids),
        )
    
if __name__ == "__main__":
    main()