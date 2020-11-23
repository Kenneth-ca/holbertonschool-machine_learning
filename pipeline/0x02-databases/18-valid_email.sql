-- Creates a trigger that resets the attribute valid_email
-- Good explanation of triggers: https://www.sitepoint.com/how-to-create-mysql-triggers/
DELIMITER $$
CREATE TRIGGER reset_email
BEFORE UPDATE ON users FOR EACH ROW
BEGIN
	IF OLD.email <> NEW.email THEN
		SET NEW.valid_email = 0;
	END IF;
END $$
DELIMITER ;
