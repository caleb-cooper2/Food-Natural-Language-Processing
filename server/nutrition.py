"""
Stage 5 of the pipeline: turns a matched FoodID plus a quantity into grams, nutrients and a density block

Recipes get resolved through their ingredient list so a composite dish adds up from its parts rather than a single row
"""

from .config import UNIT_GRAMS, RAG_TOP_N
from .known_densities import derive_density
from .resources import food_index, global_median_density, recipe_index


def resolve_grams(food_id, quantity, unit):
    """
    Converts a quantity and unit to grams for a given food entry
    :param food_id: FoodID to look up serving measures for
    :param quantity: Numeric quantity value
    :param unit: Unit string (e.g. "cup", "slice") or None
    :return: Float gram weight
    """
    if unit in UNIT_GRAMS:
        return quantity * UNIT_GRAMS[unit]

    servings = food_index.get(food_id, {}).get("serving_measure", [])

    if unit and servings:
        for serving in servings:
            label = str(serving.get("CSM") or serving.get("name") or "").lower()
            if unit.lower() in label:
                weight = (
                        serving.get("Measure")
                        or serving.get("grams")
                        or serving.get("ml")
                )
                try:
                    return quantity * float(weight)
                except (TypeError, ValueError):
                    pass

    if servings:
        weight = (
                servings[0].get("Measure")
                or servings[0].get("grams")
                or servings[0].get("ml")
        )
        try:
            return quantity * float(weight)
        except (TypeError, ValueError):
            pass

    return quantity * 100.0


def calculate_nutrients(food_id, grams):
    """
    Calculate absolute nutrient values for a food entry scaled to a given gram weight
    :param food_id: FoodID to look up
    :param grams: Gram weight to scale per-100g nutrients by
    :return: Dict of nutrient_name -> rounded float value (or None if source value is None)
    """
    entry = food_index.get(food_id)
    if not entry:
        return {}
    factor = grams / 100.0
    return {
        nutrient: round(value * factor, 2) if value is not None else None
        for nutrient, value in entry["nutrients"].items()
    }


def resolve_recipe_nutrients(food_id, grams):
    """
    Calculates nutrients for a food, blending ingredient contributions if it is a recipe
    :param food_id: FoodID to resolve (could be a recipe or direct food entry)
    :param grams: Total gram weight to calculate nutrients for
    :return: Dict of nutrient_name -> rounded float value
    """
    if food_id not in recipe_index:
        return calculate_nutrients(food_id, grams)

    total = {}
    for component in recipe_index[food_id]:
        ingredient_grams = grams * component["weight_fraction"]
        for nutrient, value in calculate_nutrients(component["ingredient_id"], ingredient_grams).items():
            if value is not None:
                total[nutrient] = round(total.get(nutrient, 0.0) + value, 2)

    return total or calculate_nutrients(food_id, grams)


def build_candidate_list(ranked, grams):
    """
    Builds the full candidate response list from ranked (food_id, score) pairs
    :param ranked: List of (food_id, score) tuples in descending score order
    :param grams: Gram weight used to calculate nutrient values
    :return: List of candidate dicts with food_id, name, score, is_recipe, recipe_ingredients, nutrients
    """
    candidates = []
    for food_id, score in ranked[:RAG_TOP_N]:
        entry = food_index.get(food_id)
        if not entry:
            continue

        display_name = entry["name"]
        brands = entry.get("brands", [])
        if food_id.startswith("OFF:") and brands:
            primary_brand = brands[0]
            if primary_brand.lower() not in display_name.lower():
                display_name = f"{primary_brand.title()} – {display_name}"

        candidates.append({
            "food_id": food_id,
            "name": display_name,
            "score": round(score, 2),
            "is_recipe": entry.get("is_recipe", False),
            "recipe_ingredients": [
                {"food_id": c["ingredient_id"], "name": c["ingredient_name"], "weight_fraction": c["weight_fraction"]}
                for c in recipe_index.get(food_id, [])
            ],
            "nutrients": resolve_recipe_nutrients(food_id, grams)
        })
    return candidates


def attach_density(match, prep):
    """
    Resolves a density for a matched candidate and attaches it in place
    :param match: The chosen candidate dict (has food_id); mutated to gain a "density" block
    :param prep: Preparation word found by the LLM (e.g. "sticks", "grated", "whole"), or None
    """
    if not match:
        return
    entry = food_index.get(match["food_id"], {})
    density, log_sigma, source = derive_density(entry, global_median_density, prep)
    match["density"] = {
        "density_g_per_ml": round(density, 4),
        "density_log_sigma": round(log_sigma, 4),
        "density_source": source
    }
