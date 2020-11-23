-- Optimize simple search
CREATE INDEX idx_name_first
ON names(name(1));
