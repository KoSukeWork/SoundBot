"""Rebuild the imported folder tree from SQLite for the renderer's sidebar.

A scan streams its folder tree to the renderer once, over the import's
WebSocket, so the sidebar lost every imported folder after a restart. This
rebuilds the same shape from the files table.
"""
import ntpath
import os
import posixpath
from typing import Any, Dict, Iterable, List, Mapping


def _is_windows_path(path: str) -> bool:
    return os.name == "nt" or bool(ntpath.splitdrive(path)[0]) or path.startswith("\\\\")


def parent_folder(path: str) -> str:
    """Return the displayed parent folder, keeping Windows paths Windows-style."""
    return ntpath.dirname(path) if _is_windows_path(path) else posixpath.dirname(path)


def _folder_name(path: str) -> str:
    name = ntpath.basename(path) if _is_windows_path(path) else posixpath.basename(path)
    return name or path


def _has_ancestor(path: str, candidates: set) -> bool:
    current = path
    while True:
        parent = parent_folder(current)
        if parent == current or not parent:
            return False
        if parent in candidates:
            return True
        current = parent


def build_folder_tree(
    folder_counts: Mapping[str, int], import_roots: Iterable[str]
) -> List[Dict[str, Any]]:
    """Return import roots as nested ``FolderNode``-shaped dicts.

    ``folder_counts`` maps each folder to the files directly inside it. Folders
    below an import root hang under it; a root nested in another root belongs
    to the outer one, and a folder outside every root becomes its own root.
    ``file_count`` includes all subfolders, as in the scan's tree.
    """
    roots = {root for root in import_roots if root}
    roots = {root for root in roots if not _has_ancestor(root, roots)}
    nodes: Dict[str, Dict[str, Any]] = {}
    top: Dict[str, Dict[str, Any]] = {}

    def node(path: str, root: str) -> Dict[str, Any]:
        existing = nodes.get(path)
        if existing is None:
            relative = path[len(root):].lstrip("\\/") if path != root else ""
            existing = nodes[path] = {
                "name": _folder_name(path),
                "path": path,
                "relative_path": relative,
                "file_count": 0,
                "children": [],
            }
        return existing

    for folder, count in folder_counts.items():
        chain = [folder]
        while chain[-1] not in roots:
            parent = parent_folder(chain[-1])
            if parent == chain[-1] or not parent:
                chain = [folder]
                break
            chain.append(parent)
        root = chain[-1]
        parent = None
        for path in reversed(chain):
            known = path in nodes
            current = node(path, root)
            if parent is None:
                top.setdefault(path, current)
            elif not known:
                parent["children"].append(current)
            parent = current
        nodes[folder]["file_count"] += int(count)

    def finish(item: Dict[str, Any]) -> int:
        item["children"].sort(key=lambda child: child["name"].casefold())
        item["file_count"] += sum(finish(child) for child in item["children"])
        return item["file_count"]

    for item in top.values():
        finish(item)
    return sorted(top.values(), key=lambda item: item["path"].casefold())
