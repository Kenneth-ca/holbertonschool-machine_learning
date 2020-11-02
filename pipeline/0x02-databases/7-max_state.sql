-- Temperatures #2
SELECT state, MAX(value) AS avg_temp
FROM temperatures
GROUP BY state
ORDER BY state ASC;
;
