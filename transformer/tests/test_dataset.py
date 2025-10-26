"""
Tests for the dataset module.
"""

import pytest
from transformer.dataset import load_and_prepare_data, clean_data, prepare_data


def test_clean_data():
    """Test data cleaning function."""
    # Mock dataset
    dataset = {"train": [{"text": "example"}]}
    cleaned = clean_data(dataset)
    
    # Should return dataset (placeholder test)
    assert cleaned is not None


def test_prepare_data():
    """Test data preparation function."""
    # Mock dataset
    dataset = {"train": [{"text": "example"}]}
    prepared = prepare_data(dataset)
    
    # Should return dataset (placeholder test)
    assert prepared is not None


# Note: Testing load_and_prepare_data requires network access to download datasets
# This can be mocked or skipped in CI/CD environments
@pytest.mark.skip(reason="Requires network access to download datasets")
def test_load_and_prepare_data():
    """Test loading data from Hugging Face."""
    dataset = load_and_prepare_data("wikitext", split="train")
    assert dataset is not None


if __name__ == "__main__":
    pytest.main([__file__])

