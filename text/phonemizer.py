"""
Phonemizer
Converts text to phonemes for multilingual TTS
Supports rule-based and dictionary-based phonemization
"""

import re
from typing import List, Dict, Optional
from .symbols import symbols, symbol_to_id, PAD_ID, BOS_ID, EOS_ID
from .normalizer import TextNormalizer


class Phonemizer:
    """Converts text to phoneme sequences"""
    
    def __init__(self, language: str = 'en', use_external: bool = False):
        """
        Initialize phonemizer
        
        Args:
            language: Language code
            use_external: Use external phonemizer library (requires phonemizer package)
        """
        self.language = language
        self.use_external = use_external
        self.normalizer = TextNormalizer(language)
        
        # Initialize phoneme mappings
        self._init_phoneme_mappings()
        
        # Try to load external phonemizer if requested
        if use_external:
            try:
                from phonemizer import phonemize
                self.external_phonemize = phonemize
            except ImportError:
                print("Warning: phonemizer package not found. Using rule-based phonemization.")
                self.use_external = False
    
    def _init_phoneme_mappings(self):
        """Initialize basic phoneme mappings for rule-based conversion"""
        # English grapheme-to-phoneme mappings (simplified)
        self.en_mappings = {
            'a': 'AH',
            'b': 'B',
            'c': 'K',
            'd': 'D',
            'e': 'EH',
            'f': 'F',
            'g': 'G',
            'h': 'HH',
            'i': 'IH',
            'j': 'JH',
            'k': 'K',
            'l': 'L',
            'm': 'M',
            'n': 'N',
            'o': 'OW',
            'p': 'P',
            'q': 'K',
            'r': 'R',
            's': 'S',
            't': 'T',
            'u': 'UH',
            'v': 'V',
            'w': 'W',
            'x': 'K S',
            'y': 'Y',
            'z': 'Z',
            'ch': 'CH',
            'sh': 'SH',
            'th': 'TH',
            'ph': 'F',
            'ng': 'NG',
        }
        
        # Common English words with known pronunciations
        self.en_dictionary = {
            'the': 'DH AH',
            'a': 'AH',
            'an': 'AE N',
            'and': 'AE N D',
            'or': 'AO R',
            'is': 'IH Z',
            'are': 'AA R',
            'was': 'W AA Z',
            'were': 'W ER',
            'be': 'B IY',
            'have': 'HH AE V',
            'has': 'HH AE Z',
            'had': 'HH AE D',
            'do': 'D UW',
            'does': 'D AH Z',
            'did': 'D IH D',
            'will': 'W IH L',
            'would': 'W UH D',
            'could': 'K UH D',
            'should': 'SH UH D',
            'can': 'K AE N',
            'may': 'M EY',
            'might': 'M AY T',
            'must': 'M AH S T',
            'to': 'T UW',
            'of': 'AH V',
            'in': 'IH N',
            'on': 'AA N',
            'at': 'AE T',
            'by': 'B AY',
            'for': 'F AO R',
            'with': 'W IH TH',
            'from': 'F R AH M',
            'about': 'AH B AW T',
            'hello': 'HH AH L OW',
            'world': 'W ER L D',
            'test': 'T EH S T',
            'speech': 'S P IY CH',
            'synthesis': 'S IH N TH AH S IH S',
        }
    
    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phoneme sequence
        
        Args:
            text: Input text
            
        Returns:
            List of phonemes
        """
        # Normalize text
        text = self.normalizer.normalize_for_phonemes(text)
        
        # Use external phonemizer if available
        if self.use_external:
            return self._external_phonemize(text)
        
        # Otherwise use rule-based phonemization
        return self._rule_based_phonemize(text)
    
    def _external_phonemize(self, text: str) -> List[str]:
        """Use external phonemizer library"""
        try:
            phonemes = self.external_phonemize(
                text,
                language=self.language,
                backend='espeak',
                strip=True,
                preserve_punctuation=True,
                with_stress=True
            )
            # Split into individual phonemes
            phonemes = phonemes.split()
            return phonemes
        except Exception as e:
            print(f"External phonemization failed: {e}. Falling back to rule-based.")
            return self._rule_based_phonemize(text)
    
    def _rule_based_phonemize(self, text: str) -> List[str]:
        """
        Rule-based phonemization (simplified)
        For production, use a proper G2P model or dictionary
        """
        phonemes = []
        words = text.split()
        
        for word in words:
            # Check if word is punctuation
            if word in '.,!?;:':
                phonemes.append(word)
                continue
            
            # Check dictionary first
            word_lower = word.lower()
            if self.language == 'en' and word_lower in self.en_dictionary:
                phonemes.extend(self.en_dictionary[word_lower].split())
            else:
                # Use grapheme-to-phoneme mapping
                word_phonemes = self._grapheme_to_phoneme(word_lower)
                phonemes.extend(word_phonemes)
            
            # Add word boundary
            phonemes.append(' ')
        
        # Remove trailing space
        if phonemes and phonemes[-1] == ' ':
            phonemes = phonemes[:-1]
        
        return phonemes
    
    def _grapheme_to_phoneme(self, word: str) -> List[str]:
        """
        Convert graphemes to phonemes using simple rules
        This is a simplified version - production systems should use proper G2P
        """
        phonemes = []
        i = 0
        
        while i < len(word):
            # Try two-character combinations first
            if i < len(word) - 1:
                bigram = word[i:i+2]
                if self.language == 'en' and bigram in self.en_mappings:
                    phonemes.append(self.en_mappings[bigram])
                    i += 2
                    continue
            
            # Single character
            char = word[i]
            if self.language == 'en' and char in self.en_mappings:
                phonemes.append(self.en_mappings[char])
            else:
                # For unknown characters, use IPA approximation
                phonemes.append(char)
            
            i += 1
        
        return phonemes
    
    def phonemes_to_ids(self, phonemes: List[str], add_bos_eos: bool = True) -> List[int]:
        """
        Convert phoneme sequence to IDs
        
        Args:
            phonemes: List of phonemes
            add_bos_eos: Add beginning/end of sequence tokens
            
        Returns:
            List of phoneme IDs
        """
        ids = []
        
        if add_bos_eos:
            ids.append(BOS_ID)
        
        for phoneme in phonemes:
            # Get ID from symbol table
            if phoneme in symbol_to_id:
                ids.append(symbol_to_id[phoneme])
            else:
                # Unknown phoneme, use PAD
                ids.append(PAD_ID)
        
        if add_bos_eos:
            ids.append(EOS_ID)
        
        return ids
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text directly to ID sequence
        
        Args:
            text: Input text
            
        Returns:
            List of phoneme IDs
        """
        phonemes = self.text_to_phonemes(text)
        ids = self.phonemes_to_ids(phonemes)
        return ids
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """
        Convert ID sequence back to text (for debugging)
        
        Args:
            sequence: List of phoneme IDs
            
        Returns:
            Text representation
        """
        from .symbols import id_to_symbol
        phonemes = [id_to_symbol.get(id, '_') for id in sequence]
        return ' '.join(phonemes)
