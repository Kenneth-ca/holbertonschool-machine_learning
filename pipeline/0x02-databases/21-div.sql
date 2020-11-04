-- Creates a function that divides
DELIMITER $$
CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS FLOAT
BEGIN
	DECLARE result FLOAT;
	SET result = 0;
	IF b <> 0 THEN
		SET result = a / b;
	END IF;
	RETURN result;
END $$
DELIMITER ;
