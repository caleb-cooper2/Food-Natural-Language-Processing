from sentence_transformers import SentenceTransformer, CrossEncoder

from .index import load_indexes
from .known_densities import compute_global_median_density
from .logging_config import get_logger

logger = get_logger("food-nlp")

food_index, recipe_index, faiss_index, faiss_ids, bm25, bm25_ids = load_indexes()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

logger.info(f"Food index: {len(food_index)} entries | Recipes: {len(recipe_index)}")
logger.info(f"FAISS index: {len(faiss_ids)} vectors")

global_median_density = compute_global_median_density(food_index)
