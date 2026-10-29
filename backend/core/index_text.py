"""Immutable lookup snapshots for indexing metadata without per-file scans."""
import re
import threading

from core.database import canonicalize_path


class FolderContext(tuple):
    """Keep the original first-match order while indexing canonical ancestors."""
    def __new__(cls, entries):
        self = super().__new__(cls, (item for item in entries if item.get('user_folder_id')))
        self.by_path = {}
        for rank, item in enumerate(self):
            key = canonicalize_path(str(item.get('folder_path') or ''))
            if key:
                self.by_path.setdefault(key.rstrip('/\\') or '/', (rank, item))
        return self

    def find(self, path):
        if not self.by_path:
            return {}
        candidate = canonicalize_path(path)
        if not candidate:
            return {}
        matches = []
        key = candidate.rstrip('/\\')
        while key:
            if key in self.by_path:
                matches.append(self.by_path[key])
            key = key.rpartition('/')[0]
        if candidate.startswith(('/', '\\')) and '/' in self.by_path:
            matches.append(self.by_path['/'])
        return min(matches, key=lambda item: item[0])[1] if matches else {}


class UCSMatcher:
    def __init__(self, mapping):
        self.categories = []
        self.keywords = []
        self.tokens, self.phrases, self.cjk = {}, {}, {}
        for rank, (category, synonyms) in enumerate(mapping.items()):
            self.categories.append(str(category))
            self.keywords.append(tuple((str(value).strip(), str(value).strip().casefold())
                                       for value in synonyms if str(value).strip()))
            for term in (category, *synonyms):
                normalized = ' '.join(t for t in re.split(r'[^0-9a-z\u3400-\u9fff]+', str(term).casefold()) if t)
                if not normalized:
                    continue
                if re.search(r'[\u3400-\u9fff]', normalized):
                    compact = normalized.replace(' ', '')
                    self.cjk.setdefault(compact[0], {}).setdefault(compact, set()).add(rank)
                elif ' ' in normalized:
                    self.phrases.setdefault(normalized.split()[0], {}).setdefault(normalized, set()).add(rank)
                else:
                    self.tokens.setdefault(normalized, set()).add(rank)

    def match(self, normalized_corpus):
        tokens = set(normalized_corpus.split())
        found = set()
        padded = f' {normalized_corpus} '
        for token in tokens:
            found.update(self.tokens.get(token, ()))
            for phrase, ranks in self.phrases.get(token, {}).items():
                if f' {phrase} ' in padded:
                    found.update(ranks)
        compact = normalized_corpus.replace(' ', '')
        for char in set(compact):
            for term, ranks in self.cjk.get(char, {}).items():
                if term in compact:
                    found.update(ranks)
        categories, keywords, seen = [], [], set()
        for rank in sorted(found):
            categories.append(self.categories[rank])
            for rendered, identity in self.keywords[rank]:
                if identity not in seen:
                    seen.add(identity)
                    keywords.append(rendered)
        return dict(categories=categories, keywords=keywords,
                    search_text=' '.join([*categories, *keywords]))


_source = _matcher = None
_lock = threading.Lock()


def ucs_matcher(mapping):
    """The loaded UCS dictionary is an immutable snapshot; reload replaces it."""
    global _source, _matcher
    with _lock:
        if mapping is not _source:
            _matcher = UCSMatcher(mapping)
            _source = mapping
        return _matcher
