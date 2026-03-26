import sys
import subprocess
import platform


def is_wsl():
    """
    Detect if running in Windows Subsystem for Linux (WSL).
    Returns True if WSL is detected, False otherwise.
    """
    return (
        "wsl" in platform.release().lower() or "microsoft" in platform.release().lower()
    )


def browse_folder():
    """
    Opens a native folder selection dialog and returns the selected folder path.
    Works on Windows, macOS, and Linux (with zenity or kdialog).
    Returns None if cancelled or error.
    """
    try:
        is_windows = sys.platform.startswith("win")
        is_wsl_env = is_wsl()
        if is_windows or is_wsl_env:
            script = (
                "Add-Type -AssemblyName System.windows.forms;"
                "$f=New-Object System.Windows.Forms.FolderBrowserDialog;"
                'if($f.ShowDialog() -eq "OK"){Write-Output $f.SelectedPath}'
            )
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=30,
            )
            folder = result.stdout.strip()
            if folder and is_wsl_env: # Convert Windows path to WSL path
                result = subprocess.run(
                    ["wslpath", "-u", folder],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                folder = result.stdout.strip()
            return folder if folder else None
        elif sys.platform == "darwin":
            script = 'POSIX path of (choose folder with prompt "Select folder:")'
            result = subprocess.run(
                ["osascript", "-e", script], capture_output=True, text=True, timeout=30
            )
            folder = result.stdout.strip()
            return folder if folder else None
        else:
            # Linux: try zenity, then kdialog
            for cmd in [
                [
                    "zenity",
                    "--file-selection",
                    "--directory",
                    "--title=Select folder",
                ],
                [
                    "kdialog",
                    "--getexistingdirectory",
                    "--title",
                    "Select folder",
                ],
            ]:
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0 or result.returncode == 1:  # zenity and kdialog return 1 on cancel
                        folder = result.stdout.strip()
                        return folder if folder else None
                except subprocess.TimeoutExpired:
                    return None
                except (FileNotFoundError, Exception):
                    continue
            return None
    except Exception:
        return None