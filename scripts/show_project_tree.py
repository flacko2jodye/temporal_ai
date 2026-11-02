from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDED = {
    ".venv",
    "data",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
}

def show_tree(directory, prefix=""):

    entries = sorted(
        [
            item for item in directory.iterdir()
            if item.name not in EXCLUDED
        ],
        key=lambda item: (item.is_file(), item.name.lower())
    )

    for index, item in enumerate(entries):

        is_last = index == len(entries) - 1
        connector = "└── " if is_last else "├── "

        print(prefix + connector + item.name)

        if item.is_dir():
            extension = "    " if is_last else "│   "
            show_tree(item, prefix + extension)


if __name__ == "__main__":
    print(ROOT.name)
    show_tree(ROOT)