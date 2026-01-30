# Concept DAS: Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions

Our paper is [accepted as Poster at ICLR 2026](https://openreview.net/forum?id=LoisXFZL3k).

Our *Concept DAS (CDAS)* is inspired by *distributed alignment search (DAS)*, which uses *distributed interchange interventions (DIIs)*.

## CDAS library (more to come)

This library is based on [axbench](https://github.com/stanfordnlp/axbench); shout out to the authors & maintainers!

Supports:
1. Training set augmentation for contrastive pairs with OpenAI API.
2. Bi-directional DII on contrastive data pairs: `(negative_x, negative_y, positive_x, positive_y)`.
3. Gather steering factors from training set.
4. Bi-directional inference.
5. Steering score evaluation with OpenAI API.
6. Standard benchmark evaluation.

Does not support:
1. Training non-DII steering vectors listed by axbench.


## Steering vectors (SVs)

The following SVs use DIIs,
whose names are tracked by `MODELS_WITH_FACTOR_FILE` of `cdas/constants.py`:

* `DASModel`, `DASVector`: Cross-entropy loss.
* `CDASModel`, `CDASVector`: Jensen-Shannon divergence loss.
* `KLDASModel`, `KLDASVector`: KL divergence loss (forward/reverse mode).
* `PDASModel`, `PDASVector`: Preference optimization objectives (SimPO/DPO loss).

