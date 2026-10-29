from pathlib import Path
import random
import re
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from core.index_text import FolderContext, UCSMatcher, ucs_matcher
from core.database import canonical_path_is_within
from core.ucs_keywords import get_ucs_keywords


def reference(mapping, corpus):
    tokens = set(corpus.split())
    def matches(term):
        value = ' '.join(t for t in re.split(r'[^0-9a-z\u3400-\u9fff]+', str(term).casefold()) if t)
        if not value:
            return False
        if re.search(r'[\u3400-\u9fff]', value):
            return value.replace(' ', '') in corpus.replace(' ', '')
        return value in tokens if ' ' not in value else f' {value} ' in f' {corpus} '
    categories, keywords, seen = [], [], set()
    for category, synonyms in mapping.items():
        if matches(category) or any(matches(value) for value in synonyms):
            categories.append(str(category))
            for value in synonyms:
                value = str(value).strip()
                if value and value.casefold() not in seen:
                    keywords.append(value);seen.add(value.casefold())
    return dict(categories=categories, keywords=keywords, search_text=' '.join([*categories,*keywords]))


class TextLookupTests(unittest.TestCase):
    def test_real_ucs_random_and_boundary_corpora_match_reference_exactly(self):
        mapping = get_ucs_keywords();matcher = UCSMatcher(mapping)
        self.assertGreater(len(mapping), 100)
        terms = [str(term) for category, synonyms in mapping.items() for term in (category,*synonyms)]
        rng = random.Random(646)
        corpora = ['', 'airplane air flow airflow', '中文 类别 中文类别', 'ß İ ﬀ café']
        for _ in range(1000):
            words = rng.choices(terms, k=rng.randint(1,5))
            raw = rng.choice([' ', '-', '_', '/', '+']).join(words)
            if rng.random()<.5:raw=raw.upper()
            if rng.random()<.3:raw='prefix'+raw+'suffix'
            corpora.append(raw)
        for raw in corpora:
            corpus=' '.join(t for t in re.split(r'[^0-9a-z\u3400-\u9fff]+',raw.casefold()) if t)
            self.assertEqual(matcher.match(corpus),reference(mapping,corpus),raw)

    def test_order_duplicates_cjk_substrings_and_reloaded_dictionary(self):
        mapping={'水 水':['air flow',' bite ','BITE','','a+b'],'水':['中文 phrase','air flow'],'air':['plane']}
        for corpus in ('bite air flow','中文phrase','airflow','prefix水水suffix','a b','plane'):
            self.assertEqual(ucs_matcher(mapping).match(corpus),reference(mapping,corpus))
        replacement={'新字典':['bite']}
        self.assertEqual(ucs_matcher(replacement).match('bite'),reference(replacement,'bite'))
        self.assertEqual(ucs_matcher(mapping).match('bite'),reference(mapping,'bite'))

    def test_folder_ancestors_boundaries_and_first_match_order(self):
        roots=['C:/audio','C:/audio/foo','C:/audio/foobar','C:/audio/foo/../foo',
               'C:/audio/foo/','C:/','//server/share','//server/share/folder',
               'C:/audio/percent%_name','/','', 'relative/folder']
        context=[dict(folder_path=p,user_folder_id=str(n+1),logical_folder=str(n)) for n,p in enumerate(roots)]
        context.insert(0,dict(folder_path='C:/audio',user_folder_id=None))
        paths=['C:/audio/foo/sound.wav','C:/audio/FOOBAR/t.wav','C:/audio/foobarish/t',
               'C:/audio/foo','C:/audio/percent%_name/t.wav','C:/else/t',
               '//server/share/folder/t.wav','//server/share2/t.wav','', '/tmp/a',
               'relative/folder/a.wav']
        rng=random.Random(24)
        for _ in range(40):
            rng.shuffle(context);lookup=FolderContext(context)
            for path in paths:
                expected=next((r for r in context if r.get('user_folder_id') and canonical_path_is_within(path,r['folder_path'])),{})
                self.assertEqual(lookup.find(path),expected,(path,context))

    def test_empty_user_folder_snapshot_skips_all_18270_mappings(self):
        context=FolderContext(dict(folder_path=f'C:/root/{n}',user_folder_id=None) for n in range(18270))
        with mock.patch('core.index_text.canonicalize_path') as canonical:
            for _ in range(100):self.assertEqual(context.find('C:/root/x/file.wav'),{})
        canonical.assert_not_called()

    def test_complete_payload_matches_unindexed_context(self):
        import main
        context=[dict(folder_path='C:/audio/foo',user_folder_id='nested',logical_folder='Water',description='wave'),
                 dict(folder_path='C:/audio',user_folder_id='root',logical_folder='Root',description='rain')]
        source=dict(path='C:/audio/foo/Water_drip.wav',filename='Water_drip.wav',metadata_tags={'tag':['水','BITE']})
        before=main._build_text_metadata_payload('default','file',source,folder_context=context)
        after=main._build_text_metadata_payload('default','file',source,folder_context=FolderContext(context))
        self.assertEqual(after,before)
