-- https://mirabelle.openfoodfacts.org/products

-- This query retrieves products from the Open Food Facts database that are available in New Zealand or Australia,
-- have been scanned at least twice, and have nutritional information available for energy, proteins, fat,
-- carbohydrates, sodium, or salt per 100g.
-- (Essentially ensuring good data quality)

-- 2964 data rows as of 21/05/2026

SELECT
    code,
    product_name,
    generic_name,
    brands,
    categories_en,
    ingredients_text,
    serving_size,
    serving_quantity,
    "energy-kj_100g",
    proteins_100g,
    fat_100g,
    carbohydrates_100g,
    fiber_100g,
    sodium_100g,
    salt_100g
FROM [all]
WHERE (countries_en LIKE '%new zealand%' OR countries_en LIKE '%australia%')
AND (
    "energy-kj_100g" IS NOT NULL
    OR proteins_100g IS NOT NULL
    OR fat_100g IS NOT NULL
    OR carbohydrates_100g IS NOT NULL
    OR sodium_100g IS NOT NULL
    OR salt_100g IS NOT NULL
)
AND serving_size IS NOT NULL
AND serving_size != ''
AND TRIM(serving_size) != ''