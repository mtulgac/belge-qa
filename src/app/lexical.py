import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

# Turkish dotted capital: "İ".lower() is "i" + U+0307 COMBINING DOT ABOVE, two
# codepoints, so "İTÜ" and "itü" would not compare equal and every proper noun
# starting with İ would silently fail to match.
COMBINING_DOT = "̇"

# The standard word split. Keeps letters and digits, drops everything else.
TOKEN_SPLIT = re.compile(r"\w+", re.UNICODE)

# The same, except a separator that sits *between* two word characters does not
# break the token.
TOKEN_KEEP = re.compile(r"\w+(?:[.,/:\-]\w+)*", re.UNICODE)

TOKEN_PATTERNS = {"split": TOKEN_SPLIT, "keep": TOKEN_KEEP}


def tokenize(text: str, pattern: re.Pattern = TOKEN_KEEP) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).lower().replace(COMBINING_DOT, "")
    return pattern.findall(folded)


class Bm25:

    def __init__(
        self,
        texts: list[str],
        pattern: re.Pattern = TOKEN_KEEP,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.pattern, self.k1, self.b = pattern, k1, b
        self.n = len(texts)

        # term -> [(document index, term frequency)]. Sparse on purpose: scoring
        # only ever walks the query's own terms, never the whole vocabulary.
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len = np.zeros(self.n, dtype=np.float64)

        for i, text in enumerate(texts):
            counts = Counter(tokenize(text, pattern))
            self.doc_len[i] = sum(counts.values())
            for term, tf in counts.items():
                self.postings[term].append((i, tf))

        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        # Robertson/Sparck-Jones IDF with the +1 that keeps it non-negative: a
        # term in every document scores 0 instead of going negative and actively
        # pushing matching documents down.
        self.idf = {
            term: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def __len__(self) -> int:
        return self.n

    def score(self, query: str) -> np.ndarray:

        scores = np.zeros(self.n, dtype=np.float64)
        if not self.n:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for term in tokenize(query, self.pattern):
            posting = self.postings.get(term)
            if not posting:
                continue
            idx = np.fromiter((i for i, _ in posting), dtype=np.int64, count=len(posting))
            tf = np.fromiter((f for _, f in posting), dtype=np.float64, count=len(posting))
            scores[idx] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[idx])
        return scores
