"""Check constrained-line legality, codec/citation alignment, and cached batching."""
import unittest
import torch
import common
from legal_lm import ConstrainedState, encode_action, generate
from nd_environment import Tokenizer
from nd_verify import parse_proof_tokens
from nd_model import ProofModel, ModelConfig
from symbolic_general_env import Action


class LegalLMTests(unittest.TestCase):
    def test_all_source_paths_roundtrip_without_reference_inputs_to_decoder(self):
        tok=Tokenizer();sources=common.read_jsonl(common.ROOT/"data/v1/sources.jsonl")
        self.assertEqual(len(sources),104)
        for source in sources:
            state=ConstrainedState({k:source[k] for k in ("source_id","prompt")},tok)
            lines=parse_proof_tokens(source["body"].split())
            while state.forced_position<len(state.forced):
                allowed=state.allowed();state.consume(allowed[0],allowed)
            while not state.done:
                allowed=state.allowed()
                if state.pending_qed:
                    state.consume(tok.eos_id,allowed);break
                expected=lines[len(state.env.lines)]
                action=Action(expected["rule"],expected["formula"],tuple(expected["refs"]),expected["depth"])
                self.assertIn(action,state.actions)
                for token in encode_action(tok,action,len(state.env.lines)+1):
                    state.consume(token,state.allowed())
            result=state.finish()
            self.assertEqual(result["reward"],1)
            self.assertLessEqual(result["n_lines"],6)
            self.assertEqual(len(result["response_ids"]),result["branch_tokens"]+result["forced_tokens"])

    def test_mask_escape_and_token_cap(self):
        tok=Tokenizer();row={"source_id":"test","prompt":"THM ( P & Q ) SEQ Q PRF"}
        state=ConstrainedState(row,tok)
        allowed=state.allowed()
        with self.assertRaises(AssertionError):state.consume(tok.stoi["<THINK>"],allowed)
        state.consume(allowed[0],allowed)
        result=state.finish()
        self.assertEqual(result["reward"],0)
        self.assertEqual(result["finish_reason"],"token_limit")

    def test_batched_cached_greedy_matches_singletons(self):
        torch.set_num_threads(1);torch.manual_seed(42);tok=Tokenizer()
        model=ProofModel(ModelConfig(vocab_size=len(tok),d_model=32,n_layers=1,n_heads=4,d_ff=64,max_seq_len=512,dropout=0)).eval()
        rows=[{"source_id":"one","prompt":"THM ( P & Q ) SEQ Q PRF"},
              {"source_id":"two","prompt":"THM SEQ ( P > P ) PRF"}]
        together=generate(model,tok,rows,max_new_tokens=128,batch_size=2,max_lines=8)
        separate=[generate(model,tok,[row],max_new_tokens=128,batch_size=1,max_lines=8)[0] for row in rows]
        for a,b in zip(together,separate):
            self.assertEqual(a["response_ids"],b["response_ids"])
            self.assertEqual(a["action_trace"],b["action_trace"])


if __name__=="__main__":unittest.main()
