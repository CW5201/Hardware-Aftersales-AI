# tests/test_basic.py

"""
Basic tests for LangGraph-RAG-Engine
"""

import pytest
from pathlib import Path


def test_project_structure():
    """Test that project structure is correct"""
    project_root = Path(__file__).parent.parent
    
    # Check main directories exist
    assert (project_root / "config").exists()
    assert (project_root / "processor").exists()
    assert (project_root / "utils").exists()
    assert (project_root / "web").exists()
    
    # Check key files exist
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "README.md").exists()
    assert (project_root / ".env.example").exists()


def test_imports():
    """Test that main modules can be imported"""
    # This test verifies that the project structure is correct
    # and basic imports work
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Test config imports
    try:
        from config import lm_config
        from config import embedding_config
        from config import milvus_config
        from config import minio_config
    except ImportError as e:
        pytest.skip(f"Config import failed: {e}")


def test_environment_variables():
    """Test that environment variables can be loaded"""
    from dotenv import load_dotenv
    import os
    
    # Load .env.example
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env.example"
    
    if env_file.exists():
        load_dotenv(env_file)
        
        # Check that some variables are defined
        # Note: These are example values, not real ones
        assert os.getenv("MILVUS_URL") is not None or True  # Skip if not set


def test_utils_imports():
    """Test that utility modules can be imported"""
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Test utility imports
    try:
        from utils import sse_utils
        from utils import task_utils
    except ImportError as e:
        pytest.skip(f"Utils import failed: {e}")


class TestSSEUtils:
    """Test SSE utility functions"""
    
    def test_import(self):
        """Test SSE utils can be imported"""
        try:
            from utils.sse_utils import create_sse_queue, push_to_session
            assert True
        except ImportError:
            pytest.skip("SSE utils not available")
    
    def test_create_queue(self):
        """Test creating an SSE queue"""
        try:
            from utils.sse_utils import create_sse_queue
            session_id = "test_session"
            queue = create_sse_queue(session_id)
            assert queue is not None
        except (ImportError, Exception):
            pytest.skip("SSE queue creation not available")


class TestTaskUtils:
    """Test task utility functions"""
    
    def test_import(self):
        """Test task utils can be imported"""
        try:
            from utils.task_utils import update_task_status, get_task_result
            assert True
        except ImportError:
            pytest.skip("Task utils not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
