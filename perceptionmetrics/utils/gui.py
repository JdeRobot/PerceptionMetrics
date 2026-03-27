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


def browse_file(filetypes=None):
    """
    Opens a native file selection dialog and returns the selected file path.
    Works on Windows, macOS, and Linux (with zenity or kdialog).

    :param filetypes: List of file extensions to filter (e.g. [".json", ".yaml"]).
                     Pass None or empty list to allow all files.
    :type filetypes: list[str] | None
    :return: Selected file path or None if cancelled.
    :rtype: str | None
    """
    try:
        is_windows = sys.platform.startswith("win")
        is_wsl_env = is_wsl()
        if is_windows or is_wsl_env:
            # Build a PowerShell filter string like "JSON files (*.json)|*.json|All files (*.*)|*.*"
            if filetypes:
                parts = []
                for ext in filetypes:
                    ext_clean = ext.lstrip(".")
                    parts.append(f"{ext_clean.upper()} files (*.{ext_clean})|*.{ext_clean}")
                parts.append("All files (*.*)|*.*")
                filter_str = "|".join(parts)
            else:
                filter_str = "All files (*.*)|*.*"

            script = (
                "Add-Type -AssemblyName System.windows.forms;"
                "$f=New-Object System.Windows.Forms.OpenFileDialog;"
                f'$f.Filter="{filter_str}";'
                'if($f.ShowDialog() -eq "OK"){Write-Output $f.FileName}'
            )
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=30,
            )
            fpath = result.stdout.strip()
            if fpath and is_wsl_env:
                result = subprocess.run(
                    ["wslpath", "-u", fpath],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                fpath = result.stdout.strip()
            return fpath if fpath else None
        elif sys.platform == "darwin":
            if filetypes:
                type_list = ", ".join(f'"{e.lstrip(".")}"' for e in filetypes)
                script = f'POSIX path of (choose file with prompt "Select file:" of type {{{type_list}}})'
            else:
                script = 'POSIX path of (choose file with prompt "Select file:")'
            result = subprocess.run(
                ["osascript", "-e", script], capture_output=True, text=True, timeout=30
            )
            fpath = result.stdout.strip()
            return fpath if fpath else None
        else:
            # Linux: try zenity, then kdialog
            for tool in ["zenity", "kdialog"]:
                try:
                    if tool == "zenity":
                        cmd = ["zenity", "--file-selection", "--title=Select file"]
                        if filetypes:
                            for ext in filetypes:
                                cmd += ["--file-filter", f"*{ext}"]
                    else:
                        cmd = ["kdialog", "--getopenfilename", "--title", "Select file"]
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=30
                    )
                    if result.returncode in (0, 1):
                        fpath = result.stdout.strip()
                        return fpath if fpath else None
                except subprocess.TimeoutExpired:
                    return None
                except (FileNotFoundError, Exception):
                    continue
            return None
    except Exception:
        return None