-- Creates a trigger when there is a new order
-- Good explanation of triggers: https://www.sitepoint.com/how-to-create-mysql-triggers/
DELIMITER $$
CREATE TRIGGER add_order
AFTER INSERT ON orders FOR EACH ROW
BEGIN
	UPDATE items
	SET quantity = quantity - NEW.number
	WHERE items.name = NEW.item_name;
END $$
DELIMITER ;
