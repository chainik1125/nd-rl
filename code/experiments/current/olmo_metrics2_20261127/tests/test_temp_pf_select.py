import base64
import sys
from pathlib import Path

import numpy as np
import pytest

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X / "tools"))
import temp_pf_select as sel  # noqa: E402

TOK = Path("/scratch/cowifornia/nd-rl/olmo_elicit/tok/dpo")


def test_decode_ids_roundtrip():
    ids = [1, 100257, 524, 27963, 29, 7]
    assert (
        sel.decode_ids(base64.b64encode(np.asarray(ids, dtype=np.int32).tobytes()).decode()) == ids
    )


@pytest.mark.skipif(not TOK.exists(), reason="tokenizer files absent")
def test_find_split_on_the_three_token_think_end():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOK))
    think = "Let me think about it carefully.\nSecond line of thought.\n"
    ans = "Final answer here."
    ids = tok.encode(think + "</think>" + ans, add_special_tokens=False)
    k = sel.find_split(tok, ids)
    assert 0 < k < len(ids)
    assert "</think>" in tok.decode(ids[:k], skip_special_tokens=True)
    assert "</think>" not in tok.decode(ids[: k - 1], skip_special_tokens=True)
    assert tok.decode(ids[k:], skip_special_tokens=True).strip().startswith("Final answer")
    ids2 = tok.encode("no closing tag here at all", add_special_tokens=False)
    assert sel.find_split(tok, ids2) == -1
