# Knowledge engine eval: `banking`

44 questions (16 ambiguous comparison questions excluded), evidence budget 8000 chars.

| system | gold-doc recall | answer in context | end-to-end correct | avg chars | avg ms |
|---|---:|---:|---:|---:|---:|
| nexus | 93% | 91% | - | 8152 | 2583 |
| hybrid | 93% | 98% | - | 7191 | 2475 |
| bm25 | 89% | 91% | - | 7021 | 3 |
| dense | 89% | 95% | - | 7193 | 2477 |

## By question type

| type | system | n | gold-doc recall | answer in context | end-to-end |
|---|---|---:|---:|---:|---:|
| single | nexus | 25 | 100% | 96% | - |
| single | hybrid | 25 | 96% | 100% | - |
| single | bm25 | 25 | 96% | 96% | - |
| single | dense | 25 | 88% | 96% | - |
| compare | nexus | 4 | 71% | 100% | - |
| compare | hybrid | 4 | 71% | 100% | - |
| compare | bm25 | 4 | 50% | 100% | - |
| compare | dense | 4 | 50% | 75% | - |
| procedure | nexus | 15 | 87% | 80% | - |
| procedure | hybrid | 15 | 93% | 93% | - |
| procedure | bm25 | 15 | 87% | 80% | - |
| procedure | dense | 15 | 100% | 100% | - |

## nexus misses (answer not in context)

- q003 (single): What is the daily ATM withdrawal limit for the Blue Account debit card?  — needs ['$500'], gold docs found 100%
- q051 (procedure): Which internal tool should an agent use to move money between a customer's checking and savings accounts, and what status must both accounts hold before the transfer can proceed?  — needs ['transfer_funds_between_bank_accounts_7291', 'ACTIVE or OPEN'], gold docs found 0%
- q059 (procedure): Which internal tool should an agent use to retrieve the debit card information for a specific checking account, and what must be verified regarding the card's status before freezing it?  — needs ['get_debit_cards_by_account_id_7823', 'Verify the card exists and check its current status'], gold docs found 0%
- q060 (procedure): Which specific tool name must an agent use to unlock and call the system when processing a replacement credit card request?  — needs ['order_replacement_credit_card_7291', 'unlock_discoverable_agent_tool', 'call_discoverable_agent_tool'], gold docs found 100%
