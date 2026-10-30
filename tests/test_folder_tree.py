from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.folder_tree import build_folder_tree, parent_folder


def walk(nodes):
    for node in nodes:
        yield node
        yield from walk(node["children"])


class FolderTreeTests(unittest.TestCase):
    def test_counts_roll_up_under_the_import_root(self) -> None:
        counts = {
            r"S:\Library\Doors": 2,
            r"S:\Library\Doors\Metal\Heavy": 3,
            r"S:\Library\wind": 1,
        }
        roots = [r"S:\Library", r"S:\Library\Doors", r"S:\Library\Doors\Metal"]

        tree = build_folder_tree(counts, roots)

        self.assertEqual([node["path"] for node in tree], [r"S:\Library"])
        root = tree[0]
        self.assertEqual((root["name"], root["relative_path"], root["file_count"]), ("Library", "", 6))
        self.assertEqual([child["name"] for child in root["children"]], ["Doors", "wind"])
        by_path = {node["path"]: node for node in walk(tree)}
        self.assertEqual(by_path[r"S:\Library\Doors"]["file_count"], 5)
        # A folder with no files of its own still links its subfolders.
        metal = by_path[r"S:\Library\Doors\Metal"]
        self.assertEqual((metal["file_count"], metal["relative_path"]), (3, r"Doors\Metal"))
        self.assertEqual(by_path[r"S:\Library\Doors\Metal\Heavy"]["file_count"], 3)
        self.assertEqual(len(by_path), 5)

    def test_folders_outside_every_root_become_roots(self) -> None:
        tree = build_folder_tree({r"D:\Loose\Clips": 4, r"S:\Library\Rain": 1}, [r"S:\Library"])

        self.assertEqual([(node["path"], node["file_count"]) for node in tree],
                         [(r"D:\Loose\Clips", 4), (r"S:\Library", 1)])
        self.assertEqual(tree[0]["children"], [])

    def test_drive_root_import_and_posix_paths(self) -> None:
        tree = build_folder_tree({"S:\\Pack": 1}, ["S:\\"])
        self.assertEqual((tree[0]["name"], tree[0]["file_count"]), ("S:\\", 1))
        self.assertEqual(tree[0]["children"][0]["relative_path"], "Pack")

        self.assertEqual(parent_folder(r"S:\Library\a.wav"), r"S:\Library")
        tree = build_folder_tree({"/Volumes/SFX/Foley": 2}, ["/Volumes/SFX"])
        self.assertEqual(tree[0]["children"][0]["relative_path"], "Foley")

    def test_empty_library(self) -> None:
        self.assertEqual(build_folder_tree({}, [r"S:\Library"]), [])


if __name__ == "__main__":
    unittest.main()
