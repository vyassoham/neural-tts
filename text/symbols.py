"""
Phoneme symbols and mappings for multilingual TTS
Supports IPA (International Phonetic Alphabet) and language-specific phonemes
"""

# Special symbols
_pad = '_'
_eos = '~'
_bos = '^'

# Punctuation
_punctuation = '!\'(),.:;? '

# English phonemes (ARPAbet-style)
_english_phonemes = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
]

# IPA consonants (covering most languages)
_ipa_consonants = [
    'p', 'b', 't', 'd', 'k', 'g', 'ʔ',  # Plosives
    'm', 'n', 'ɲ', 'ŋ',  # Nasals
    'r', 'ɾ', 'ʀ', 'ʁ',  # Trills/taps
    'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'ʂ', 'ʐ', 'ç', 'ʝ', 'x', 'ɣ', 'χ', 'ʁ', 'h', 'ɦ',  # Fricatives
    'l', 'ɫ', 'ɬ', 'ɮ',  # Laterals
    'w', 'ɥ', 'ʋ', 'ɹ', 'ɻ', 'j', 'ɰ',  # Approximants
    'ʦ', 'ʣ', 'ʧ', 'ʤ',  # Affricates
]

# IPA vowels (covering most languages)
_ipa_vowels = [
    'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',  # Close vowels
    'ɪ', 'ʏ', 'ʊ',  # Near-close vowels
    'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',  # Close-mid vowels
    'ə',  # Mid vowel (schwa)
    'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ',  # Open-mid vowels
    'æ', 'ɐ',  # Near-open vowels
    'a', 'ɶ', 'ɑ', 'ɒ',  # Open vowels
]

# Diacritics and suprasegmentals
_ipa_diacritics = [
    'ː',  # Long
    'ˑ',  # Half-long
    '̆',  # Extra-short
    'ˈ',  # Primary stress
    'ˌ',  # Secondary stress
    '.',  # Syllable break
    '|',  # Minor break
    '‖',  # Major break
]

# Tones (for tonal languages like Chinese, Thai, Vietnamese)
_tones = ['˥', '˦', '˧', '˨', '˩', '˥˩', '˩˥', '˧˥', '˨˩']

# Combine all symbols
symbols = (
    [_pad, _bos, _eos] +
    list(_punctuation) +
    _english_phonemes +
    _ipa_consonants +
    _ipa_vowels +
    _ipa_diacritics +
    _tones
)

# Remove duplicates while preserving order
seen = set()
symbols = [x for x in symbols if not (x in seen or seen.add(x))]

# Create mappings
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Special token IDs
PAD_ID = symbol_to_id[_pad]
BOS_ID = symbol_to_id[_bos]
EOS_ID = symbol_to_id[_eos]


def get_symbol_id(symbol):
    """Get ID for a symbol, return PAD_ID if not found"""
    return symbol_to_id.get(symbol, PAD_ID)


def get_symbol(symbol_id):
    """Get symbol for an ID"""
    return id_to_symbol.get(symbol_id, _pad)
