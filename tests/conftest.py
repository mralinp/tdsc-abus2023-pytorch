import pytest
import os
import tempfile

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_dataset_path(temp_data_dir):
    """Create a sample dataset structure."""
    # You can add setup code here to create sample data
    return temp_data_dir 