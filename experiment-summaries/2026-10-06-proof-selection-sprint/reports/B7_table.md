| set | m (dev target) | count-stop + siblings | count-stop + trained judge | sibling threshold (phase 5) | trained-judge threshold | oracle: Lean stops at first valid (at the trained-judge threshold's attempts) |
|---|---|---|---|---|---|---|
| holdout250 | 2 | 0.889 @ 1.59 (c=1) | 0.889 @ 1.59 (c=1) | 0.889 @ 1.59 | **0.889 @ 1.59** | 0.941 |
| holdout250 | 4 | 0.915 @ 2.79 (c=2) | 0.920 @ 2.79 (c=2) | 0.929 @ 3.85 | **0.935 @ 3.12** | 0.953 |
| holdout250 | 8 | 0.933 @ 7.09 (c=6) | 0.938 @ 7.09 (c=6) | 0.937 @ 8.22 | **0.941 @ 6.45** | 0.953 |
| textbook72 | 2 | 0.527 @ 3.99 (c=1) | 0.527 @ 3.99 (c=1) | 0.527 @ 3.99 | **0.527 @ 3.99** | 0.599 |
| textbook72 | 4 | 0.567 @ 6.08 (c=2) | 0.579 @ 6.08 (c=2) | 0.611 @ 9.83 | **0.630 @ 9.14** | 0.653 |
| textbook72 | 8 | 0.604 @ 10.75 (c=6) | 0.624 @ 10.75 (c=6) | 0.613 @ 13.40 | **0.633 @ 12.05** | 0.653 |
