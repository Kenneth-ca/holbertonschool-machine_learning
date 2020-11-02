-- Number of shows by genre
SELECT tg.name AS genre, COUNT(g.genre_id) AS number_of_shows
FROM tv_show_genres AS g
JOIN tv_genres AS tg
ON tg.id = g.genre_id
GROUP BY genre
ORDER BY number_of_shows DESC
;
