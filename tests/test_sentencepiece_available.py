"""Test that sentencepiece is available and has required functionality."""

def test_sentencepiece_import():
    """Test that sentencepiece can be imported."""
    import sentencepiece
    assert sentencepiece is not None


def test_sentencepiece_trainer_available():
    """Test that SentencePieceTrainer is available."""
    import sentencepiece
    assert hasattr(sentencepiece, 'SentencePieceTrainer')
