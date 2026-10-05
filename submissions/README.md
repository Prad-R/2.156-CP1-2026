# Best submission so far

`best_submission.npy` is the best-scoring submission found so far, in the course's submission format. It is updated and pushed automatically whenever a merge improves any of the three problems (see `publish_best.py`). `best_submission.json` records its score from `evaluate_submission`, the number of designs per problem, and which run each problem's designs came from.

Get the latest copy from any branch:

```bash
git fetch origin
git checkout origin/prads-branch -- submissions/
```

Load and score it:

```python
import numpy as np
from LINKS.CP import evaluate_submission

submission = np.load("submissions/best_submission.npy", allow_pickle=True).item()
print(evaluate_submission(submission))
```
