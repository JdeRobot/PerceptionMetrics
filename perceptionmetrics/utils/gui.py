import tkinter as tk
from tkinter import filedialog


def browse_folder():
    """
    Opens  folder selection dialog and returns the selected folder path.

    :returns: Absolute path of the selected folder, or ``None`` if the
              dialog is cancelled or an error occurs.
    :rtype: str | None

    """
    result = None

    try:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        # Bring dialog to the front
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(
            title="Select folder",
            mustexist=True,
        )
        if folder:
            result = folder
        root.destroy()
        return result
    except Exception:
        return None
