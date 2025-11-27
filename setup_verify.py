"""
Quick Setup and Test Script
Verifies installation and runs a simple test
"""

import sys
import os


def check_dependencies():
    """Check if all required packages are installed"""
    print("=" * 60)
    print("Checking Dependencies...")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'numpy': 'NumPy',
        'librosa': 'Librosa',
        'soundfile': 'SoundFile',
        'tqdm': 'tqdm'
    }
    
    missing = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20s} - Installed")
        except ImportError:
            print(f"✗ {name:20s} - Missing")
            missing.append(package)
    
    if missing:
        print("\n" + "=" * 60)
        print("Missing packages detected!")
        print("=" * 60)
        print("\nPlease install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All required packages are installed!")
    return True


def check_pytorch():
    """Check PyTorch installation and CUDA availability"""
    print("\n" + "=" * 60)
    print("PyTorch Configuration")
    print("=" * 60)
    
    import torch
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\nNote: CUDA not available. Training will be slow on CPU.")
        print("For GPU support, install PyTorch with CUDA:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")


def verify_project_structure():
    """Verify all required files and directories exist"""
    print("\n" + "=" * 60)
    print("Verifying Project Structure")
    print("=" * 60)
    
    required_dirs = [
        'config',
        'text',
        'models',
        'models/acoustic',
        'models/vocoder',
        'models/speaker',
        'data',
        'training',
        'inference',
        'utils',
        'examples'
    ]
    
    required_files = [
        'requirements.txt',
        'README.md',
        'config/acoustic_config.py',
        'config/vocoder_config.py',
        'text/phonemizer.py',
        'models/acoustic/fastspeech2.py',
        'models/vocoder/hifigan.py',
        'training/train_acoustic.py',
        'inference/synthesizer.py'
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ Directory: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            all_good = False
    
    # Check files
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✓ File: {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
            all_good = False
    
    if all_good:
        print("\n✓ Project structure is complete!")
    else:
        print("\n✗ Some files or directories are missing!")
    
    return all_good


def test_imports():
    """Test importing main modules"""
    print("\n" + "=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    modules_to_test = [
        ('config.acoustic_config', 'AcousticConfig'),
        ('config.vocoder_config', 'VocoderConfig'),
        ('text.phonemizer', 'Phonemizer'),
        ('models.acoustic.fastspeech2', 'FastSpeech2'),
        ('models.vocoder.hifigan', 'HiFiGANGenerator'),
        ('data.preprocessing', 'AudioPreprocessor'),
    ]
    
    all_good = True
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"✗ {module_name}.{class_name} - Error: {e}")
            all_good = False
    
    if all_good:
        print("\n✓ All modules imported successfully!")
    else:
        print("\n✗ Some modules failed to import!")
    
    return all_good


def create_directories():
    """Create necessary directories"""
    print("\n" + "=" * 60)
    print("Creating Directories")
    print("=" * 60)
    
    dirs_to_create = [
        'checkpoints',
        'checkpoints/acoustic',
        'checkpoints/vocoder',
        'checkpoints/speaker',
        'outputs',
        'dataset',
        'dataset/wavs'
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    print("\n✓ All directories created!")


def run_simple_test():
    """Run a simple test of the text processing pipeline"""
    print("\n" + "=" * 60)
    print("Running Simple Test")
    print("=" * 60)
    
    try:
        from text.phonemizer import Phonemizer
        from text.normalizer import TextNormalizer
        
        # Test normalizer
        normalizer = TextNormalizer('en')
        text = "Dr. Smith said it's 5:30 PM."
        normalized = normalizer.normalize(text)
        print(f"\nOriginal: {text}")
        print(f"Normalized: {normalized}")
        
        # Test phonemizer
        phonemizer = Phonemizer('en')
        phonemes = phonemizer.text_to_phonemes("Hello world")
        print(f"\nText: Hello world")
        print(f"Phonemes: {phonemes}")
        
        # Test sequence conversion
        sequence = phonemizer.text_to_sequence("Hello")
        print(f"\nText: Hello")
        print(f"Sequence: {sequence}")
        
        print("\n✓ Text processing test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup and verification"""
    print("\n" + "=" * 60)
    print("Neural TTS System - Setup and Verification")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠ Please install missing dependencies first!")
        return
    
    # Check PyTorch
    check_pytorch()
    
    # Verify project structure
    if not verify_project_structure():
        print("\n⚠ Project structure is incomplete!")
        return
    
    # Create directories
    create_directories()
    
    # Test imports
    if not test_imports():
        print("\n⚠ Some modules failed to import!")
        return
    
    # Run simple test
    if not run_simple_test():
        print("\n⚠ Simple test failed!")
        return
    
    # Success!
    print("\n" + "=" * 60)
    print("✓ Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Read TRAINING_GUIDE.md to prepare your dataset")
    print("2. Train the acoustic model: python training/train_acoustic.py --help")
    print("3. Train the vocoder: python training/train_vocoder.py --help")
    print("4. Run inference examples: python examples/example_synthesis.py")
    print("\nFor detailed documentation, see:")
    print("- README.md - Project overview")
    print("- ARCHITECTURE.md - Technical details")
    print("- TRAINING_GUIDE.md - Training instructions")
    print("- INFERENCE_GUIDE.md - Usage guide")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
