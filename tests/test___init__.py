from unittest.mock import patch
from datanavigator import _check_ffmpeg, _check_clip_folder

@patch("shutil.which")
@patch("builtins.print")
def test_check_ffmpeg_installed(mock_print, mock_which):
    # Simulate ffmpeg being available in PATH
    mock_which.return_value = "/usr/bin/ffmpeg"
    _check_ffmpeg()
    mock_print.assert_not_called()

@patch("shutil.which")
@patch("builtins.print")
def test_check_ffmpeg_not_installed(mock_print, mock_which):
    # Simulate ffmpeg not being available in PATH
    mock_which.return_value = None
    with patch("sys.platform", "win32"):
        _check_ffmpeg()
        mock_print.assert_any_call("\nFFmpeg is not installed or not in PATH.")
        mock_print.assert_any_call("Download it from: https://ffmpeg.org/download.html")

@patch("os.path.exists")
@patch("builtins.print")
def test_check_clip_folder_exists(mock_print, mock_path_exists):
    # Simulate clip folder already existing
    mock_path_exists.return_value = True
    _check_clip_folder()
    mock_print.assert_not_called()
