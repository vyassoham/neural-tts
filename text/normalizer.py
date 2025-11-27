"""
Text Normalizer
Handles text cleaning, Unicode normalization, and preprocessing for multilingual support
"""

import re
import unicodedata
from typing import Dict, List


class TextNormalizer:
    """Normalizes text for TTS processing"""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize text normalizer
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr', 'zh')
        """
        self.language = language
        self._init_language_rules()
    
    def _init_language_rules(self):
        """Initialize language-specific normalization rules"""
        # Number words for different languages
        self.number_words = {
            'en': ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
            'es': ['cero', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve'],
            'fr': ['zéro', 'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf'],
            'de': ['null', 'eins', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 'sieben', 'acht', 'neun'],
            'zh': ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'],
        }
        
        # Common abbreviations
        self.abbreviations = {
            'en': {
                'mr.': 'mister',
                'mrs.': 'misess',
                'dr.': 'doctor',
                'st.': 'saint',
                'co.': 'company',
                'jr.': 'junior',
                'sr.': 'senior',
                'etc.': 'et cetera',
            },
            'es': {
                'sr.': 'señor',
                'sra.': 'señora',
                'dr.': 'doctor',
                'dra.': 'doctora',
            },
            'fr': {
                'm.': 'monsieur',
                'mme.': 'madame',
                'dr.': 'docteur',
            },
        }
    
    def normalize(self, text: str) -> str:
        """
        Normalize text through multiple stages
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Expand abbreviations
        text = self._expand_abbreviations(text)
        
        # Normalize numbers
        text = self._normalize_numbers(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Clean special characters (keep only letters, numbers, and basic punctuation)
        text = self._clean_text(text)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations"""
        if self.language in self.abbreviations:
            for abbr, expansion in self.abbreviations[self.language].items():
                text = re.sub(r'\b' + re.escape(abbr), expansion, text, flags=re.IGNORECASE)
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        """Convert numbers to words"""
        def replace_number(match):
            num = match.group(0)
            try:
                # Handle simple single digits
                if len(num) == 1:
                    digit = int(num)
                    if self.language in self.number_words:
                        return self.number_words[self.language][digit]
                # For multi-digit numbers, keep as is or use more complex conversion
                # This is a simplified version
                return num
            except:
                return num
        
        # Replace standalone numbers
        text = re.sub(r'\b\d\b', replace_number, text)
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters"""
        # Keep letters (all Unicode letters), numbers, spaces, and basic punctuation
        # This preserves characters from all languages
        text = re.sub(r'[^\w\s\'\-,.:;!?]', '', text, flags=re.UNICODE)
        return text
    
    def normalize_for_phonemes(self, text: str) -> str:
        """
        Additional normalization specifically for phoneme conversion
        
        Args:
            text: Input text
            
        Returns:
            Text ready for phonemization
        """
        text = self.normalize(text)
        
        # Remove punctuation that doesn't affect pronunciation
        # Keep only sentence-ending punctuation and pauses
        text = re.sub(r'[^\w\s.,!?;:]', '', text, flags=re.UNICODE)
        
        return text
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    @staticmethod
    def add_punctuation_pauses(text: str) -> str:
        """
        Add pause markers for punctuation
        
        Args:
            text: Input text
            
        Returns:
            Text with pause markers
        """
        # Add pauses after punctuation
        text = re.sub(r'([,;:])', r'\1 |', text)  # Short pause
        text = re.sub(r'([.!?])', r'\1 ‖', text)  # Long pause
        return text
