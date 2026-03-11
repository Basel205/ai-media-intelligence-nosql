import os


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def get_all_image_paths(root_folder: str) -> list[str]:
    """Recursively collect all image file paths under root_folder."""
    paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS:
                paths.append(os.path.join(dirpath, fname))
    return paths


def is_image(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() in SUPPORTED_EXTENSIONS
