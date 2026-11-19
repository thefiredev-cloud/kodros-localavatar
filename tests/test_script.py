from pathlib import Path

import pytest

from localavatar.script import MAX_CHARS, normalize, sentences, split_script

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "proof-script.txt"


def words(text: str) -> list[str]:
    return text.split()


def test_sentences_split_on_terminal_punctuation():
    assert sentences("One thing. Two things! Three? Done.") == ["One thing.", "Two things!", "Three?", "Done."]


def test_abbreviation_like_lowercase_does_not_split():
    # "e.g. a" has no capital after the period, so it stays one sentence.
    assert sentences("Use a model, e.g. a small one. Next.") == ["Use a model, e.g. a small one.", "Next."]


def test_every_word_preserved_in_order():
    text = FIXTURE.read_text()
    chunks = split_script(text)
    assert words(" ".join(c.text for c in chunks)) == words(normalize(text))


def test_paragraph_ends_marked():
    chunks = split_script("First para one is a sentence. First para two is longer than that.\n\nSecond paragraph sentence here.")
    assert [c.text for c in chunks] == [
        "First para one is a sentence.",
        "First para two is longer than that.",
        "Second paragraph sentence here.",
    ]
    assert [c.paragraph_end for c in chunks] == [False, True, True]


def test_long_sentence_split_under_limit():
    long = ("This clause goes on, " * 30).strip().rstrip(",") + "."
    chunks = split_script(long)
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHARS for c in chunks)
    assert words(" ".join(c.text for c in chunks)) == words(long)


def test_long_clause_without_punctuation_split_at_words():
    long = " ".join(["word"] * 120) + "."
    chunks = split_script(long, max_chars=60)
    assert all(len(c.text) <= 60 for c in chunks)
    assert len(words(" ".join(c.text for c in chunks))) == 120


def test_short_sentences_merged():
    chunks = split_script("Hi. Okay. This is a normal length sentence that stands alone.")
    assert len(chunks) < 3
    assert chunks[0].text.startswith("Hi. Okay.")


def test_empty_and_whitespace():
    assert split_script("") == []
    assert split_script(" \n\n \t") == []


def test_curly_quotes_normalized():
    assert normalize("“Hello” it’s") == "\"Hello\" it's"


def test_rejects_tiny_limit():
    with pytest.raises(ValueError):
        split_script("text", max_chars=5)
