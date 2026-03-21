import sys
import subprocess


def browse_folder():
    """
    Opens a native folder selection dialog and returns the selected folder path.
    Works on Windows, macOS, and Linux (with zenity or kdialog).
    Returns None if cancelled or error.
    """
    try:
        if sys.platform.startswith("win"):
            script = (
                "Add-Type -AssemblyName System.windows.forms;"
                "$f=New-Object System.Windows.Forms.FolderBrowserDialog;"
                'if($f.ShowDialog() -eq "OK"){Write-Output $f.SelectedPath}'
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
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
                    folder = result.stdout.strip()
                    if folder:
                        return folder
                except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                    continue
            return None
    except Exception:
        return None
