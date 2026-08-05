"""Generic TF-IDF retrieval index over a list of JSONL-style dict records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalIndex:
    documents: List[Dict]
    vectorizer: TfidfVectorizer
    matrix: "np.ndarray | object"  # scipy sparse matrix

    def query(self, text: str, k: int = 3) -> List[Dict]:
        if not self.documents:
            return []
        vec = self.vectorizer.transform([text])
        sims = cosine_similarity(vec, self.matrix)[0]
        top_idx = np.argsort(-sims)[:k]
        return [self.documents[i] for i in top_idx]


def build_index(documents: List[Dict], text_fn: Callable[[Dict], str]) -> RetrievalIndex:
    texts = [text_fn(d) or "" for d in documents]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return RetrievalIndex(documents=documents, vectorizer=vectorizer, matrix=matrix)
