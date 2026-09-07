from pathlib import Path

from minibert.data.wikitext import _article_title, iter_wikitext_documents


def test_article_title_detects_only_top_level_titles() -> None:
    assert _article_title("= First article =") == "First article"
    assert _article_title("= = Section heading = =") is None
    assert _article_title("Ordinary text.") is None

def test_iter_wikitext_document_groups_article(tmp_path: Path) -> None:
    input_path = tmp_path / "wiki.tokens"
    input_path.write_text(
        "\n".join(
            [
                "= First article =",
                "",
                "First article body.",
                "= = First section = =",
                "More body text.",
                "",
                "= Second article =",
                "A second article.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert list(iter_wikitext_documents(
        input_path.read_text(encoding="utf-8").splitlines()
    )) == [
        "First article First article body. = = First section = = More body text.",
        "Second article A second article."
    ]

def test_iter_wikitext_documents_handles_file_without_title(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "plain.tokens"
    input_path.write_text("A plain document.\nWith another line.\n", encoding="utf-8")

    assert list(iter_wikitext_documents(
        input_path.read_text(encoding="utf=8").splitlines()
    )) == [
        "A plain document. With another line."
    ]