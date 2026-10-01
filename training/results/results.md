Test set: 5,760 movies the model never saw during training or tuning (train 26,877, validation 5,760). Chosen C = 1.0.

| Genre | Test movies | Threshold | Precision | Recall | F1 | F1 at 0.5 | F1 always-yes |
|---|--:|--:|--:|--:|--:|--:|--:|
| Action | 1,032 | 0.55 | 0.577 | 0.721 | **0.641** | 0.634 | 0.304 |
| Adventure | 936 | 0.56 | 0.530 | 0.660 | **0.588** | 0.587 | 0.280 |
| Comedy | 1,913 | 0.48 | 0.604 | 0.741 | **0.666** | 0.665 | 0.499 |
| Crime | 777 | 0.57 | 0.515 | 0.644 | **0.572** | 0.578 | 0.238 |
| Drama | 3,022 | 0.39 | 0.700 | 0.870 | **0.776** | 0.755 | 0.688 |
| Family | 529 | 0.65 | 0.632 | 0.603 | **0.617** | 0.570 | 0.168 |
| Fantasy | 347 | 0.66 | 0.434 | 0.427 | **0.430** | 0.457 | 0.114 |
| Horror | 610 | 0.62 | 0.741 | 0.754 | **0.747** | 0.734 | 0.192 |
| Musical | 381 | 0.66 | 0.367 | 0.388 | **0.378** | 0.380 | 0.124 |
| Mystery | 300 | 0.69 | 0.403 | 0.443 | **0.422** | 0.395 | 0.099 |
| Romance | 1,133 | 0.47 | 0.468 | 0.721 | **0.568** | 0.572 | 0.329 |
| Science Fiction | 343 | 0.58 | 0.594 | 0.735 | **0.657** | 0.642 | 0.112 |
| Thriller | 1,043 | 0.50 | 0.497 | 0.677 | **0.573** | 0.573 | 0.307 |
| War | 232 | 0.72 | 0.546 | 0.638 | **0.588** | 0.590 | 0.077 |
| Western | 153 | 0.65 | 0.775 | 0.719 | **0.746** | 0.682 | 0.052 |
| *micro average* | | | 0.582 | 0.718 | **0.643** | 0.624 | 0.257 |
| *macro average* | | | 0.559 | 0.649 | **0.598** | 0.588 | 0.239 |

- 95% bootstrap interval: micro-F1 0.636 to 0.649, macro-F1 0.588 to 0.608
- Per-movie (samples) F1: 0.631. Exact genre set right: 17.0% of movies. No genre predicted: 0.7% of movies.
- Export check: on all 86,400 test (movie, genre) pairs, the exported model.json gives probabilities within 4.1e-08 of scikit-learn's, and 0 yes/no decisions differ.
