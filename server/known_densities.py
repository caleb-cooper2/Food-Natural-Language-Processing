"""
Density priors derived for a matched food, so a volume estimate can be eventually turned into a mass

Sources are tried strongest first (a density measured against the food itself, then one implied by a grams/mL serving
pair, then the global median) and each carries its own uncertainty, widening as the source weakens
"""

import math
import statistics

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger

logger = get_logger("food-priors")

# Relative (coefficient-of-variation) uncertainty by density source. These widen as the source weakens
MEASURED_DENSITY_CV = 0.10 # a density recorded against the food itself
SPREAD_FLOOR_CV = 0.10 # smallest CV used when several density rows agree closely
RATIO_DENSITY_CV = 0.15 # density implied by a single grams/cm3 serving pair
FALLBACK_DENSITY_CV = 0.30 # global-median fallback

MIN_DENSITY = 0.1
MAX_DENSITY = 2.0


def relative_to_log_sigma(relative_sigma):
    """
    Converts a relative (coefficient of variation) uncertainty into a log-space sigma
    :param relative_sigma: Fractional uncertainty, e.g. 0.10 for +/-10%
    :return: Equivalent sigma in natural-log space
    """
    return math.sqrt(math.log(1.0 + relative_sigma ** 2))


def is_possible(density):
    """Returns True if a density is within a bound that makes sense"""
    return density is not None and MIN_DENSITY <= density <= MAX_DENSITY


def measured_values(entry):
    """
    Returns the plausible density values from an entry's labelled density list
    :param entry: Food index entry with a measured_densities list of {value, prep}
    :return: List of plausible density floats, possibly empty
    """
    return [
        record["value"]
        for record in entry.get("measured_densities", []) if is_possible(record.get("value"))
    ]


def reduce_measured(values):
    """
    Reduces several measured densities to a point estimate and a spread-based relative sigma
    :param values: List of plausible density floats
    :return: Tuple of (median_density, relative_sigma)
    """
    median = statistics.median(values)
    if len(values) >= 2:
        # A real spread across preparations widens sigma
        relative_sigma = max((max(values) - min(values)) / (2 * median), SPREAD_FLOOR_CV)
    else:
        relative_sigma = MEASURED_DENSITY_CV
    return median, relative_sigma


def ratio_densities(entry):
    """
    Yields every grams-per-millilitre ratio implied by the food's serving measures
    :param entry: Food index entry with a serving_measure list
    :return: List of plausible g/mL ratios, possibly empty
    """
    ratios = []
    for measure in entry.get("serving_measure", []):
        grams = measure.get("grams") or measure.get("Measure")
        millilitres = measure.get("ml")
        if grams and millilitres:
            ratio = grams / millilitres
            if is_possible(ratio):
                ratios.append(ratio)
    return ratios


def select_by_presentation(entry, prep):
    """
    Picks the stored density whose descriptor label contains the LLM's prep word
    :param entry: Food index entry with a measured_densities list of {value, prep}
    :param prep: Preparation word from the LLM, or None
    :return: (density, relative_sigma) for the matched row, or None if no row matches
    """
    if not prep:
        return None
    for record in entry.get("measured_densities", []):
        if is_possible(record.get("value")) and prep in record.get("prep", ""):
            return record["value"], MEASURED_DENSITY_CV
    return None


def derive_density(entry, global_median_density, prep=None):
    """
    Estimates a food's density and its log-space uncertainty from the best source available
    :param entry: Food index entry dict
    :param global_median_density: Median density used as the final fallback
    :param prep: Preparation word from the LLM ("sticks", "grated", "whole"), or None
    :return: (density_g_per_ml, log_sigma, source), source in {"measured_presentation", "measured", "ratio", "global_median"}
    """
    values = measured_values(entry)

    if values:
        # If prep matches one of the food's own density rows -> use that row
        matched = select_by_presentation(entry, prep)
        if matched is not None:
            density, relative_sigma = matched
            return density, relative_to_log_sigma(relative_sigma), "measured_presentation"

        # Otherwise no usable prep, use median
        density, relative_sigma = reduce_measured(values)
        return density, relative_to_log_sigma(relative_sigma), "measured"

    ratios = ratio_densities(entry)
    if ratios:
        density = statistics.median(ratios)
        if len(ratios) >= 3:
            relative_sigma = max(statistics.stdev(ratios) / density, RATIO_DENSITY_CV)
        else:
            relative_sigma = RATIO_DENSITY_CV
        return density, relative_to_log_sigma(relative_sigma), "ratio"

    return global_median_density, relative_to_log_sigma(FALLBACK_DENSITY_CV), "global_median"


def compute_global_median_density(food_index):
    """
    Computes the median density across every food that has a measured or ratio-derived density
    :param food_index: Combined NZ/AUS/OFF food index dict
    :return: Median density in g/cm3
    """
    densities = []
    for entry in food_index.values():
        values = measured_values(entry)
        if values:
            densities.append(statistics.median(values))
            continue
        ratios = ratio_densities(entry)
        if ratios:
            densities.append(statistics.median(ratios))

    median = statistics.median(densities)
    logger.info(f"Global median density found {median:.3f} g/cm3 from {len(densities)} foods with data")
    return median