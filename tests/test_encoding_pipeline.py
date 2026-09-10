import json
from pathlib import Path

from minibert.data.corpus import iter_corpus_texts
from minibert.data.encoding import iter_encoded_documents
from minibert.data.token_storage import TokenizedCorpus, write_tokenized_corpus
from minibert.tokenizer.bert import BertTokenizer
from minibert.tokenizer.vocabulary import SPECIAL_TOKENS, Vocabulary


def test_encoding_pipeline_round_trip(tmp_path: Path) -> None:
    input_path = tmp_path / "corpus.jsonl"
    output_dir = tmp_path / "encoded"

    texts = ["Hello world", "", "World hello"]

    input_path.write_text(
        "".join(
            json.dumps({"text": text}) + "\n"
            for text in texts
        ),
        encoding="utf-8",
    )

    vocabulary = Vocabulary(
        [*SPECIAL_TOKENS, "hello", "world"]
    )
    tokenizer = BertTokenizer(vocabulary)

    documents = iter_encoded_documents(
        iter_corpus_texts(input_path),
        tokenizer,
    )
    write_tokenized_corpus(documents, output_dir, vocabulary)

    corpus = TokenizedCorpus(output_dir)

    assert len(corpus) == 2
    assert corpus.num_tokens == 4
    assert corpus[0].tolist() == [5, 6]
    assert corpus[1].tolist() == [6, 5]
    assert corpus.vocabulary.tokens == vocabulary.tokens