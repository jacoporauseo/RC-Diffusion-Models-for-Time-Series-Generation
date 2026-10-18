from pathlib import Path

def print_tree(root=".", prefix="", ignore={".git", "__pycache__", ".ipynb_checkpoints", ".venv", "venv", "node_modules"}):
    root = Path(root)
    entries = sorted([e for e in root.iterdir() if e.name not in ignore],
                      key=lambda e: (e.is_file(), e.name.lower()))
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry.name)
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, prefix + extension, ignore)

print_tree(".")