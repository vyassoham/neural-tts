"""
Neural TTS - Production-grade Text-to-Speech System
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "Production-grade neural TTS with extreme clarity, voice cloning, and multilingual support"

from inference.synthesizer import TTSSynthesizer
from inference.voice_cloner import VoiceCloner

# Simple API for easy usage
class TTS:
    """Simplified TTS API for easy daily use"""
    
    def __init__(self, model_path=None):
        """
        Initialize TTS with automatic model loading
        
        Args:
            model_path: Path to model directory (None = download from hub)
        """
        if model_path is None:
            # Auto-download models from HuggingFace or similar
            model_path = self._download_models()
        
        self.synthesizer = TTSSynthesizer(
            acoustic_checkpoint=f"{model_path}/acoustic_model.pt",
            vocoder_checkpoint=f"{model_path}/vocoder_model.pt"
        )
    
    def _download_models(self):
        """Download pre-trained models"""
        import os
        from pathlib import Path
        
        # Create cache directory
        cache_dir = Path.home() / ".cache" / "neural_tts"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # TODO: Implement model download from HuggingFace Hub
        # For now, return local path
        return "checkpoints"
    
    def speak(self, text, language='en', output_file=None, **kwargs):
        """
        Convert text to speech (simple API)
        
        Args:
            text: Text to synthesize
            language: Language code (default: 'en')
            output_file: Save to file (None = return audio array)
            **kwargs: pitch_scale, speed_scale, energy_scale
            
        Returns:
            audio array if output_file is None, else None
        """
        audio = self.synthesizer.synthesize(text, language=language, **kwargs)
        
        if output_file:
            self.synthesizer.save_audio(audio, output_file)
            return None
        
        return audio
    
    def __call__(self, text, **kwargs):
        """Allow TTS() to be called directly"""
        return self.speak(text, **kwargs)


# Convenience function
def speak(text, language='en', output_file='output.wav', **kwargs):
    """
    Quick one-liner to convert text to speech
    
    Usage:
        >>> from neural_tts import speak
        >>> speak("Hello, world!")
    """
    tts = TTS()
    return tts.speak(text, language=language, output_file=output_file, **kwargs)


__all__ = [
    'TTS',
    'speak',
    'TTSSynthesizer',
    'VoiceCloner',
    '__version__'
]
