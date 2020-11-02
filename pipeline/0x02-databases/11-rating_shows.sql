-- Rotten tomatoes
SELECT s.title, SUM(r.rate) AS rating
FROM tv_show_ratings as r
JOIN tv_shows AS s
ON r.show_id = s.id
GROUP BY s.title
ORDER BY rating DESC
;
