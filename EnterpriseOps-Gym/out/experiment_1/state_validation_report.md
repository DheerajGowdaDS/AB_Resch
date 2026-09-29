# Phase 2 - state-model validation

Anchor -> Route -> Tables -> Records -> Delivery -> Precision -> Recall.
Precision/recall are evaluator-only initial-state prompt/seed metrics and never feed the agent.

| Task | Anchor | Route | Tables | Records | Delivery | Precision | Recall |
|---|---|---|---|---|---|---|---|
| 1c0_8c7a6205 | product:128 | UNRESOLVED | 4 | 32 | yes | 0.031 | 0.050 |
| 1c0_8c7a6205 | product:128 | UNRESOLVED | 4 | 32 | yes | 0.031 | 0.050 |
| 1c0_8c7a6205 | product:128 | UNRESOLVED | 4 | 32 | yes | 0.031 | 0.050 |
| 3c3_fce76046 | location:692 | UNRESOLVED | 2 | 2 | yes | 0.500 | 1.000 |
| 3c3_fce76046 | location:692 | UNRESOLVED | 2 | 2 | yes | 0.500 | 1.000 |
| 3c3_fce76046 | location:692 | UNRESOLVED | 2 | 2 | yes | 0.500 | 1.000 |
| e0b_f075229a | product:121 | COMPLETE | 7 | 32 | yes | 0.031 | 0.083 |
| e0b_f075229a | product:121 | COMPLETE | 7 | 32 | yes | 0.031 | 0.083 |
| e0b_f075229a | product:121 | COMPLETE | 7 | 32 | yes | 0.031 | 0.083 |
| 3c3_47b48a98 | customer_case:888 | COMPLETE | 4 | 13 | yes | 0.154 | 0.400 |
| 3c3_47b48a98 | customer_case:888 | COMPLETE | 4 | 13 | yes | 0.154 | 0.400 |
| 3c3_47b48a98 | customer_case:888 | COMPLETE | 4 | 13 | yes | 0.154 | 0.400 |
| 1c0_0a53db6c | user_group:18 | COMPLETE | 5 | 77 | yes | 0.013 | 0.250 |
| 1c0_0a53db6c | user_group:18 | COMPLETE | 5 | 77 | yes | 0.013 | 0.250 |
| 1c0_0a53db6c | user_group:18 | COMPLETE | 5 | 77 | yes | 0.013 | 0.250 |
| 1c0_7304e48e | account:8 | COMPLETE | 4 | 39 | yes | 0.026 | 0.077 |
| 1c0_7304e48e | account:8 | COMPLETE | 4 | 39 | yes | 0.026 | 0.077 |
| 1c0_7304e48e | account:8 | COMPLETE | 4 | 39 | yes | 0.026 | 0.077 |
| 7e3_3757206c | account:7 | COMPLETE | 4 | 45 | yes | 0.000 | 0.000 |
| 7e3_3757206c | account:7 | COMPLETE | 4 | 45 | yes | 0.000 | 0.000 |
| 7e3_3757206c | account:7 | COMPLETE | 4 | 45 | yes | 0.000 | 0.000 |
| 84d_910b176a | product:17 | COMPLETE | 5 | 9 | yes | 0.111 | 0.062 |
| 84d_910b176a | product:17 | COMPLETE | 5 | 9 | yes | 0.111 | 0.062 |
| 84d_910b176a | product:17 | COMPLETE | 5 | 9 | yes | 0.111 | 0.062 |
| 7e3_7221f3c8 | customer_case:57 | COMPLETE | 4 | 19 | yes | 0.000 | 0.000 |
| 7e3_7221f3c8 | customer_case:57 | COMPLETE | 4 | 19 | yes | 0.000 | 0.000 |
| 7e3_7221f3c8 | customer_case:57 | COMPLETE | 4 | 19 | yes | 0.000 | 0.000 |
| 7e3_5480e26f | user_group:11 | COMPLETE | 7 | 17 | yes | 0.059 | 0.048 |
| 7e3_5480e26f | user_group:11 | COMPLETE | 7 | 17 | yes | 0.059 | 0.048 |
| 7e3_5480e26f | user_group:11 | COMPLETE | 7 | 17 | yes | 0.059 | 0.048 |
| ece_a86667c0 | user_group:44 | COMPLETE | 5 | 26 | yes | 0.038 | 0.053 |
| ece_a86667c0 | user_group:44 | COMPLETE | 5 | 26 | yes | 0.038 | 0.053 |
| ece_a86667c0 | user_group:44 | COMPLETE | 5 | 26 | yes | 0.038 | 0.053 |
