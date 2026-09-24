import platform
import subprocess
import sys

import streamlit as st
from streamlit_image_select import image_select


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
            if folder and is_wsl_env:
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
                    if result.returncode == 0 or result.returncode == 1:
                        folder = result.stdout.strip()
                        return folder if folder else None
                except subprocess.TimeoutExpired:
                    return None
                except (FileNotFoundError, Exception):
                    continue
            return None
    except Exception:
        return None


def browse_file():
    """
    Opens a native file selection dialog and returns the selected file path.
    Works on Windows, macOS, and Linux (with zenity or kdialog).
    Returns None if cancelled or error.
    """
    try:
        is_windows = sys.platform.startswith("win")
        is_wsl_env = is_wsl()
        if is_windows or is_wsl_env:
            script = (
                "Add-Type -AssemblyName System.windows.forms;"
                "$f=New-Object System.Windows.Forms.OpenFileDialog;"
                'if($f.ShowDialog() -eq "OK"){Write-Output $f.FileName}'
            )
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=30,
            )
            file_path = result.stdout.strip()
            if file_path and is_wsl_env:
                result = subprocess.run(
                    ["wslpath", "-u", file_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                file_path = result.stdout.strip()
            return file_path if file_path else None
        elif sys.platform == "darwin":
            script = 'POSIX path of (choose file with prompt "Select file:")'
            result = subprocess.run(
                ["osascript", "-e", script], capture_output=True, text=True, timeout=30
            )
            file_path = result.stdout.strip()
            return file_path if file_path else None
        else:
            for cmd in [
                ["zenity", "--file-selection", "--title=Select file"],
                ["kdialog", "--getopenfilename", "--title", "Select file"],
            ]:
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0 or result.returncode == 1:
                        file_path = result.stdout.strip()
                        return file_path if file_path else None
                except subprocess.TimeoutExpired:
                    return None
                except (FileNotFoundError, Exception):
                    continue
            return None
    except Exception:
        return None


def render_image_grid(
    item_names,
    image_paths,
    state_prefix,
    context,
    search_label="image",
    images_per_page=12,
):
    total_pages = (len(item_names) + images_per_page - 1) // images_per_page
    page_key = f"{state_prefix}_page_{context}"

    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    current_page = max(0, min(st.session_state[page_key], total_pages - 1))
    st.session_state[page_key] = current_page

    start_idx = current_page * images_per_page
    page_item_names = item_names[start_idx : start_idx + images_per_page]
    page_image_paths = image_paths[start_idx : start_idx + images_per_page]

    col1, col2, col3, col4 = st.columns([0.5, 9.5, 0.5, 0.5])
    with col1:
        if st.button(
            "⟨",
            key=f"{state_prefix}_prev_page_btn",
            disabled=(current_page == 0),
        ):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    with col2:
        st.markdown(
            f"<div style='text-align:center;font-weight:bold;'>Page {current_page + 1} of {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with col3:
        if st.button(
            "⟩",
            key=f"{state_prefix}_next_page_btn",
            disabled=(current_page >= total_pages - 1),
        ):
            st.session_state[page_key] = current_page + 1
            st.rerun()
    with col4:
        if st.button(
            "🔍",
            key=f"{state_prefix}_search_icon_btn",
            help=f"Search for a {search_label} by name",
        ):
            st.session_state[f"show_{state_prefix}_search"] = True

    if st.session_state.get(f"show_{state_prefix}_search", False):
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            selected_item = st.selectbox(
                f"Search {search_label}:",
                options=item_names,
                key=f"{state_prefix}_search_item",
            )
        with col2:
            st.markdown(
                "<div style='margin-bottom: 2.4rem;'></div>",
                unsafe_allow_html=True,
            )
            if st.button(f"Go to {search_label}", key=f"{state_prefix}_go_to_item"):
                selected_idx = item_names.index(selected_item)
                new_page = selected_idx // images_per_page
                st.session_state[page_key] = new_page
                st.session_state[f"{state_prefix}_select_{context}_{new_page}"] = (
                    selected_idx % images_per_page
                )
                st.session_state[f"show_{state_prefix}_search"] = False
                st.rerun()
        with col3:
            st.markdown(
                "<div style='margin-bottom: 2.4rem;'></div>",
                unsafe_allow_html=True,
            )
            if st.button("Cancel", key=f"{state_prefix}_cancel_search"):
                st.session_state[f"show_{state_prefix}_search"] = False
                st.rerun()

    caption_len_limit = 17
    captions = [
        (
            (name[:caption_len_limit] + "..." + name[-3:])
            if len(name) > caption_len_limit
            else name
        )
        for name in page_item_names
    ]

    select_key = f"{state_prefix}_select_{context}_{current_page}"
    select_index = st.session_state.get(select_key)
    if select_index is None or not isinstance(select_index, int):
        select_index = 0

    selected_image_path = (
        image_select(
            label="",
            images=page_image_paths,
            captions=captions,
            use_container_width=False,
            key=select_key,
            index=select_index,
        )
        if page_image_paths
        else None
    )

    if not selected_image_path:
        return None, None

    selected_index = page_image_paths.index(selected_image_path)
    return selected_image_path, page_item_names[selected_index]
