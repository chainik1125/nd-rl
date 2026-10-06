| set | policy | m ≈ 2 | m ≈ 4 | m ≈ 8 |
|---|---|---|---|---|
| holdout250 | pick one proof at random (fixed k) | 0.829 [0.818-0.836] @ 2.00 | 0.829 [0.818-0.836] @ 4.00 | 0.829 [0.818-0.836] @ 8.00 |
| holdout250 | pick a proof the environment marked finished (fixed k) | 0.870 [0.856-0.877] @ 2.00 | 0.883 [0.864-0.893] @ 4.00 | 0.889 [0.870-0.900] @ 8.00 |
| holdout250 | stop at the first finished proof | 0.892 [0.872-0.904] @ 1.59 (closest reachable) | 0.892 [0.872-0.904] @ 1.59 (closest reachable) | 0.892 [0.872-0.904] @ 1.59 (closest reachable) |
| holdout250 | stop after c finished; the model keeps its own most-likely one | 0.892 [0.872-0.904] @ 1.59 (closest reachable) | 0.914 [0.901-0.921] @ 3.92 | 0.921 [0.911-0.928] @ 8.11 |
| holdout250 | stop after c finished; two sibling checkpoints judge | 0.892 [0.872-0.904] @ 1.59 (closest reachable) | 0.924 [0.912-0.932] @ 3.92 | 0.934 [0.926-0.942] @ 8.11 |
| holdout250 | oracle: any valid proof among k (pass@k) | 0.898 [0.889-0.904] @ 2.00 | 0.927 [0.919-0.934] @ 4.00 | 0.943 [0.935-0.955] @ 8.00 |
| holdout250 | oracle: stop at the first valid proof | 0.952 [0.941-0.966] @ 1.98 | 0.953 [0.941-0.968] @ 2.03 (closest reachable) | 0.953 [0.941-0.968] @ 2.03 (closest reachable) |
| textbook72 | pick one proof at random (fixed k) | 0.427 [0.407-0.439] @ 2.00 | 0.427 [0.407-0.439] @ 4.00 | 0.427 [0.407-0.439] @ 8.00 |
| textbook72 | pick a proof the environment marked finished (fixed k) | 0.471 [0.456-0.488] @ 2.00 | 0.501 [0.493-0.516] @ 4.00 | 0.517 [0.508-0.531] @ 8.00 |
| textbook72 | stop at the first finished proof | 0.501 [0.493-0.516] @ 2.04 | 0.527 [0.518-0.537] @ 3.96 | 0.527 [0.518-0.537] @ 3.96 (closest reachable) |
| textbook72 | stop after c finished; the model keeps its own most-likely one | 0.527 [0.518-0.537] @ 3.96 (closest reachable) | 0.527 [0.518-0.537] @ 3.96 | 0.568 [0.553-0.592] @ 7.60 |
| textbook72 | stop after c finished; two sibling checkpoints judge | 0.527 [0.518-0.537] @ 3.96 (closest reachable) | 0.527 [0.518-0.537] @ 3.96 | 0.587 [0.571-0.612] @ 7.60 |
| textbook72 | oracle: any valid proof among k (pass@k) | 0.498 [0.477-0.523] @ 2.00 | 0.558 [0.539-0.595] @ 4.00 | 0.607 [0.577-0.653] @ 8.00 |
| textbook72 | oracle: stop at the first valid proof | 0.534 [0.515-0.566] @ 2.07 | 0.599 [0.572-0.643] @ 3.82 | 0.652 [0.620-0.703] @ 7.18 |
