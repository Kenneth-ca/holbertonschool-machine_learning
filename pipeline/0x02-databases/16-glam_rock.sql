-- Ranks Glam rock bands by longevity
SELECT band_name, IF(split IS NULL, YEAR(CURDATE()), split) - formed AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC
;
