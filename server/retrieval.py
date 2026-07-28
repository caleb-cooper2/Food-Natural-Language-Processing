"""
Stage 3 of the pipeline, the named entity linking: takes a food description and comes back with a ranked list of potential
FOODfiles/AUSNUT/OFF candidates

Retrieval combines three signals using RRF (spaCy PhraseMatcher, FAISS semantic search, BM25 lexical search), then
rank_candidates rescores that shortlist with a blend of lexical/semantic
"""

import numpy as np
import spacy
from rapidfuzz import fuzz
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

from .index import extract_brand_keywords, simple_singular, term_variants
from .logging_config import get_logger
from .resources import bm25, bm25_ids, embedding_model, faiss_index, faiss_ids, food_index

logger = get_logger("food-nlp")


# spaCy FOOD entity tagging

def build_food_matcher(nlp, food_index):
    """
    Builds a PhaseMatcher and term-to-candidate mapping from the food index
    :param nlp: spaCy language model used to create phrase docs
    :param food_index: Primary food index dict
    :return: Tuple of (matcher, term_candidates) where term_candidates maps term -> list of FoodIDs
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    term_candidates = {}

    for food_id, entry in food_index.items():
        if food_id.startswith("OFF:"):
            terms = list(dict.fromkeys(
                [entry["key_term"]] + entry.get("brands", [])
            ))
        else:
            terms = list(dict.fromkeys([entry["key_term"]] + entry["keywords"]))

        for term in terms:
            if len(term) <= 3:
                continue
            for variant in term_variants(term):
                term_candidates.setdefault(variant, []).append(food_id)

    for term, food_ids in term_candidates.items():
        ranked = sorted(list(dict.fromkeys(food_ids)), key=lambda food_id: len(food_index[food_id]["keywords"]))
        term_candidates[term] = ranked
        try:
            matcher.add(ranked[0], [nlp.make_doc(term)])
        except Exception:
            pass

    return matcher, term_candidates


class FoodNERComponent:
    """Custom spaCy pipeline component that tags FOOD entities using a PhraseMatcher"""

    def __init__(self, nlp, food_index, matcher, term_candidates):
        self.food_index = food_index
        self.matcher = matcher
        self.term_candidates = term_candidates

    def __call__(self, doc):
        """
        Tags FOOD spans and attach candidate FoodID lists as span extensions
        :param doc: spaCy Doc to annotate
        :return: Annotated Doc with FOOD entities and _.candidates extension set
        """
        spans = []
        for match_id, start, end in self.matcher(doc):
            food_id = doc.vocab.strings[match_id]
            span = Span(doc, start, end, label="FOOD")
            span._.food_id = food_id
            term = doc[start:end].text.lower()
            candidates = (
                self.term_candidates.get(term)
                or self.term_candidates.get(simple_singular(term))
                or [food_id]
            )
            span._.candidates = candidates
            spans.append(span)

        doc.ents = spacy.util.filter_spans(list(doc.ents) + spans)
        return doc


@Language.factory("food_ner")
def create_food_ner(nlp, name): # spaCy passes this automatically, have to leave name unused as a result
    """
    spaCy factory that instantiates FoodNERComponent for the pipeline
    :param nlp: spaCy language model (injected by spaCy)
    :param name: Component name string (injected by spaCy)
    :return: FoodNERComponent instance
    """
    matcher, term_candidates = build_food_matcher(nlp, food_index)
    return FoodNERComponent(nlp, food_index, matcher, term_candidates)


if not Span.has_extension("food_id"):
    Span.set_extension("food_id", default=None)
if not Span.has_extension("candidates"):
    Span.set_extension("candidates", default=[])

nlp = spacy.load("en_core_web_md", disable=["ner"])
nlp.add_pipe("food_ner", last=True)


# Individual retrieval signals

def candidate_scorer(query, candidate_key_term):
    """
    Scores a query string against a single candidate term using fuzzy matching with a specificity penalty
    :param query: User's food description string
    :param candidate_key_term: Key term from a food index entry
    :return: Float score in range [0, 108] (exact match returning 100.0)
    """
    query_lower = query.lower().strip()
    candidate_lower = candidate_key_term.lower().strip()

    if query_lower == candidate_lower:
        return 100.0

    query_tokens = set(query_lower.split())
    candidate_tokens = set(candidate_lower.split())

    precision_score = fuzz.token_sort_ratio(query_lower, candidate_lower)
    partial_score = fuzz.token_sort_ratio(query_lower, candidate_lower)

    base_score = 0.65 * precision_score + 0.35 * partial_score

    extra_tokens = candidate_tokens - query_tokens
    specificity_penalty = (len(extra_tokens) / max(len(candidate_tokens), 1)) * 15

    length_ratio = min(len(query_lower), len(candidate_lower)) / max(len(query_lower), len(candidate_lower))
    length_bonus = length_ratio * 8

    return max(0.0, base_score - specificity_penalty + length_bonus)


def semantic_search(query, k=10):
    """
    Search the FAISS index for the k most semantically similar food entries
    :param query: Food description string to encode and search
    :param k: Number of nearest neighbours to return
    :return: List of (food_id, cosine_similarity_score) tuples
    """
    vec = embedding_model.encode([query])
    vec = vec / np.linalg.norm(vec)
    scores, indices = faiss_index.search(vec.astype(np.float32), k)
    return [(faiss_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0])]


def fuzzy_search(span_text, threshold=60, limit=10):
    """
    Search the food index by fuzzy key_term matching, used as a retrieval fallback
    :param span_text: Food description string to match against
    :param threshold: Minimum candidate_scorer score to include a result
    :param limit: Maximum number of results to return
    :return: List of (food_id, score) tuples sorted by decending score
    """
    span_lower = span_text.lower().strip()
    scored = []

    for food_id, entry in food_index.items():
        key_term = entry["key_term"]
        score = candidate_scorer(span_lower, key_term)

        if score >= threshold:
            scored.append((food_id, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:limit]


def bm25_search(query, k=20):
    """
    Searches the BM25 index for the k highest-scoring food entries
    :param query: Food description string to tokenise and score
    :param k: Number of top results to return
    :return: List of (food_id, bm25_score) tuples for entries with score > 0
    """
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(bm25_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]


def reciprocal_rank_fusion(*ranked_lists, k=60):
    """
    Join multiple ranked candidate lists into a single ranking using Reciprocal Rank Fusion.
    Score for each candidate is sum(1/(k+rank)) across all lists
    :param ranked_lists: Variable number of lists of (food_id, score) tuples
    :param k: RRF smoothing constant
    :return: List of (food_id, rrf_score) tuples sorted by descending score
    """
    scores = {}
    for ranked in ranked_lists:
        for rank, (food_id, _) in enumerate(ranked):
            scores[food_id] = scores.get(food_id, 0.0) + 1.0 / (k + rank+1)
    return sorted(scores.items(), key=lambda x: -x[1])


# Retrieval and ranking

def retrieve_candidates(food_description, limit=50):
    """
    Retrieve a ranked list of candidate FoodIDs for a food description using RRF over three signals.
    Joining PhraseMatcher (spaCy NER), FAISS semantic search and BM25 lexical search.
    Fuzzy search fills the remaining slots if the fused list is shorter than the limit.
    :param food_description: Food description string to retrieve candidates for
    :param limit: Maximum number of candidate FoodIDs to return
    :return: List of FoodID strings
    """
    doc = nlp(food_description)
    phrase_ids = []
    seen = set()

    for entity in doc.ents:
        if entity.label_ == "FOOD":
            for cid in entity._.candidates:
                if cid not in seen:
                    seen.add(cid)
                    phrase_ids.append(cid)
    phrase_ranked = [(fid, 1.0) for fid in phrase_ids]

    semantic_ranked = semantic_search(food_description, k=20)

    bm25_ranked = bm25_search(food_description, k=20)

    fused = reciprocal_rank_fusion(phrase_ranked, semantic_ranked, bm25_ranked)

    fused_ids = {fid for fid, _ in fused}
    if len(fused) < limit:
        fuzzy_hits = fuzzy_search(food_description, threshold=55, limit=20)
        for fid, _ in fuzzy_hits:
            if fid not in fused_ids:
                fused.append((fid, 0.0))
                fused_ids.add(fid)

    return [fid for fid, _ in fused[:limit]]


def rank_candidates(span_text, candidate_ids, limit = 10):
    """
    Score and rank a list of candidate FoodIDs against a query using lexical and semantic signals
    :param span_text: Food description string to score against
    :param candidate_ids: List of FoodIDs to rank
    :param limit: Maximum number of ranked results to return
    :return: List of (food_id, score) tuples sorted by descending score, scaled to [0, 150]
    """
    span_lower   = span_text.lower()
    semantic_map = dict(semantic_search(span_text, k=10))
    scored = []

    for food_id in candidate_ids:
        entry = food_index.get(food_id)
        if not entry:
            continue

        candidate_terms = list(dict.fromkeys(
            [entry["key_term"]] + entry.get("keywords", [])[:5]
        ))
        lexical = max(
            candidate_scorer(span_lower, term) for term in candidate_terms
        ) / 100.0

        semantic = semantic_map.get(food_id, 0.0)

        semantic_weight = 0.4 * min(1.0, max(0.0, (semantic - 0.3) / 0.4))
        score = (1.0 - semantic_weight) * lexical + semantic_weight * semantic

        q_toks = len(span_lower.split())
        score += 0.25 * any(b in span_lower for b in entry.get("brands", [])) # bonus for brands

        if q_toks <= 3: # penalty for short multi-token queries
            key_toks = len(entry["key_term"].split())
            extra = max(0, key_toks - q_toks)
            score -= extra * 0.04

        score += 0.05 * (q_toks >= 2 and food_id in semantic_map) # bonus for long tokens that are semantically matching

        candidate_key_lower = entry["key_term"].lower()
        if span_lower in candidate_key_lower or candidate_key_lower in span_lower:
            score += 0.15 # bonus for substring matches

        scored.append((food_id, max(0.0, min(1.5, score)) * 100))

    return sorted(scored, key=lambda x: -x[1])[:limit]


def compute_confidence(ranked_scores):
    """
    Compute a confidence score for the top-ranked match based on its score and gap to second place
    :param ranked_scores: List of float scores in descending order
    :return: Float confidence in range [0.0, 1.0]
    """
    if not ranked_scores:
        return 0.0
    if len(ranked_scores) == 1:
        return ranked_scores[0] / 100.0

    gap = ranked_scores[0] - ranked_scores[1]
    return min(1.0, (ranked_scores[0] / 100.0) * (1 + gap / 100.0))


def validate_llm_items(llm_items):
    """
    Enrich LLM-extracted items with link confidence scores, falling back to brand stripping if needed
    :param llm_items: list of grounded LLM item dicts
    :return: List of item dicts with link_confidence and optional food_generic fields added
    """
    enriched = []

    for item in llm_items:
        food_name = item["food"]

        candidate_ids = retrieve_candidates(food_name)
        ranked = rank_candidates(food_name, candidate_ids)

        if ranked:
            ranked_scores = [score for _, score in ranked]
            item["link_confidence"] = compute_confidence(ranked_scores)
        else:
            item["link_confidence"] = 0.0

        # If confidence is low, try stripping brand names and re-linking
        if item["link_confidence"] < 0.4:
            top_id = ranked[0][0] if ranked else ""
            entry = food_index.get(top_id, {})
            brands = (
                entry.get("brands", [])
                if top_id.startswith("OFF:")
                else extract_brand_keywords(food_name, nlp=nlp)
            )

            for brand in brands:
                generic_attempt = food_name.replace(brand, "").strip()

                if not generic_attempt:
                    continue

                candidate_ids = retrieve_candidates(generic_attempt)
                ranked = rank_candidates(generic_attempt, candidate_ids)

                if ranked:
                    score = ranked[0][1] / 100.0
                    if score > item["link_confidence"]:
                        item["food_generic"] = generic_attempt
                        item["link_confidence"] = score

        enriched.append(item)

    return enriched
