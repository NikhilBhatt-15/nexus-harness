---
name: refunds
description: Checklist for processing a customer return or refund request
---
1. Look up the customer and the order in the database (check tier, status, delivered_on).
2. Ask the knowledge base for the return window and refund method for that tier.
3. Today's date decides the window: compare it with delivered_on.
4. Apply exactly the documented database changes, then re-query to verify them.
5. Report what you changed, with the amounts.
