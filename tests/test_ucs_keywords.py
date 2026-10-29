from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / 'backend'
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core import ucs_keywords


class UCSKeywordTests(unittest.TestCase):
    def setUp(self) -> None:
        ucs_keywords._ucs_keywords_cache = None
        ucs_keywords._ucs_loaded = False

    def test_committed_xlsx_loads_without_pandas_or_openpyxl(self) -> None:
        mapping = ucs_keywords.load_ucs_keywords(
            str(REPO_ROOT / 'UCS+音效分类中英文对照表.xlsx')
        )

        self.assertIn('气体', mapping)
        self.assertIn('compressed air', mapping['气体'])
        self.assertIn('航空器', mapping)
        self.assertGreater(len(mapping), 100)

    def test_cached_matching_keeps_exact_labels_keyword_order_and_phrase_boundaries(self):
        import main
        mapping = {
            '中文 类别': ['air-flow', 'CHOMP', 'air flow'],
            '食物': ['bite', 'CHOMP', 'bite'],
            '不应匹配': ['bit', 'airplane'],
            '字面标点': ['a+b'],
        }
        with mock.patch('core.ucs_keywords.get_ucs_keywords', return_value=mapping):
            for _ in range(3):
                result = main._derive_ucs_metadata(['中文类别', {'tag': 'BITE'}, ['a+b', 'air flow']])
                self.assertEqual(result, {
                    'categories': ['中文 类别', '食物', '字面标点'],
                    'keywords': ['air-flow', 'CHOMP', 'air flow', 'bite', 'a+b'],
                    'search_text': '中文 类别 食物 字面标点 air-flow CHOMP air flow bite a+b',
                })


if __name__ == '__main__':
    unittest.main()
