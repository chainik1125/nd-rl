"""ws3 evaluation package (plan T3): binary reward and the evaluator.

Importable from anywhere with `code/ws3` on sys.path (see `ws3/ndpath.py`), so every track scores
proofs the same way:

    import ndpath
    from eval import reward, verify_body, evaluate_pool, evaluate_textbook, next_token_accuracy
"""

from eval.evaluator import (  # noqa: F401
    UNKNOWN_LENGTH,
    aggregate,
    evaluate_pool,
    evaluate_textbook,
    generate,
    judge,
    next_token_accuracy,
    textbook_table,
    utc_now,
    wilson,
)
from eval.reward import ProofReward, is_unparseable, reward, verify_body  # noqa: F401
