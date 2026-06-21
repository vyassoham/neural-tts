import pytest
from text.normalizer import TextNormalizer
from text.phonemizer import Phonemizer


def test_text_normalizer_abbreviations():
    normalizer = TextNormalizer('en')
    text = "Hello Dr. Smith and Mr. Jones."
    normalized = normalizer.normalize(text)
    assert "doctor" in normalized
    assert "mister" in normalized
    assert "dr." not in normalized
    assert "mr." not in normalized


def test_text_normalizer_numbers():
    normalizer = TextNormalizer('en')
    # Single digits are normalized to words
    assert normalizer.normalize("I have 5 apples") == "i have five apples"
    assert normalizer.normalize("There are 9 trees") == "there are nine trees"
    
    # Multi-digits remain as numbers
    assert normalizer.normalize("Room 101") == "room 101"


def test_text_normalizer_clean():
    normalizer = TextNormalizer('en')
    text = "Hello @world! #TTS"
    normalized = normalizer.normalize(text)
    assert "@" not in normalized
    assert "#" not in normalized
    assert "hello world! tts" in normalized


def test_sentence_splitting():
    sentences = TextNormalizer.split_sentences("First sentence. Second sentence! Third one?")
    assert len(sentences) == 3
    assert sentences[0] == "First sentence"
    assert sentences[1] == "Second sentence"
    assert sentences[2] == "Third one"


def test_phonemizer_rule_based():
    phonemizer = Phonemizer('en', use_external=False)
    # Check simple vocabulary mapping
    phonemes = phonemizer.text_to_phonemes("hello world")
    assert phonemes == ['HH', 'AH', 'L', 'OW', ' ', 'W', 'ER', 'L', 'D']


def test_phonemizer_sequence():
    phonemizer = Phonemizer('en', use_external=False)
    sequence = phonemizer.text_to_sequence("hello")
    assert isinstance(sequence, list)
    assert len(sequence) > 0
    assert all(isinstance(x, int) for x in sequence)


def test_sequence_to_text():
    phonemizer = Phonemizer('en', use_external=False)
    sequence = phonemizer.text_to_sequence("hello")
    text_representation = phonemizer.sequence_to_text(sequence)
    assert isinstance(text_representation, str)
    assert len(text_representation) > 0
