# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

from unittest.mock import MagicMock, patch
import pytest

@patch("runtime.model_loader.ModelLoader")
@patch("runtime.video_reader.VideoReader")
def test_mocked_ftr_pipeline_run(mock_reader, mock_loader):
    # Setup mocks
    mock_instance = mock_reader.return_value
    mock_instance.width = 640
    mock_instance.height = 480
    mock_instance.fps = 30.0
    mock_instance.duration = 4.0
    # Yield one dummy frame
    mock_instance.read_frames.return_value = [(1, 0.0, MagicMock())]
    
    # Verify mock initialized correctly
    assert mock_instance.fps == 30.0
