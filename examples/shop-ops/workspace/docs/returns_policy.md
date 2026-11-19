# Returns and Refunds Policy

## Return window
- Standard customers can return delivered items within 14 days of the delivery date.
- Gold tier customers can return delivered items within 30 days of the delivery date.
- Items that were never delivered (cancelled or still shipping) are not returns; cancel them instead.

## Refund method
1. Confirm the order is delivered and inside the customer's return window.
2. Refund the full order total as store credit by inserting a row into `credits` (customer_id, amount, reason).
3. Set the order status to `returned`.
4. Use the reason text `return:<order id>`.

## Outside the window
- Do not refund. Offer a 10% goodwill credit only to gold tier customers, with reason `goodwill:<order id>`.
