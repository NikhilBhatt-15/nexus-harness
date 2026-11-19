"""Create the demo shop database (workspace/shop.db)."""
import sqlite3
from pathlib import Path

db = Path(__file__).parent / "workspace" / "shop.db"
db.unlink(missing_ok=True)
con = sqlite3.connect(db)
con.executescript("""
CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, email TEXT, tier TEXT);
CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, item TEXT, total REAL, status TEXT, ordered_on TEXT, delivered_on TEXT);
CREATE TABLE credits(id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id INTEGER, amount REAL, reason TEXT);
INSERT INTO customers VALUES
 (1,'Alice Rao','alice@example.com','gold'),(2,'Bob Singh','bob@example.com','standard'),
 (3,'Chen Wei','chen@example.com','gold'),(4,'Dana Iyer','dana@example.com','standard');
INSERT INTO orders VALUES
 (101,1,'Espresso machine',420.00,'delivered','2026-10-01','2026-10-05'),
 (102,1,'Coffee beans',35.50,'delivered','2026-11-02','2026-11-04'),
 (103,2,'Blender',89.99,'delivered','2026-11-08','2026-11-10'),
 (104,3,'Stand mixer',310.00,'delivered','2026-09-10','2026-09-14'),
 (105,3,'Kettle',49.00,'cancelled','2026-11-10',NULL),
 (106,4,'Toaster',59.00,'delivered','2026-11-12','2026-11-15'),
 (107,2,'Knife set',129.00,'shipped','2026-11-15',NULL);
""")
con.commit()
print("created", db)
