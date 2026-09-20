"""Independent CPU-only integration checks; invented data and launch mocks only."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
sys.path.insert(0, str(HERE))
import torch
import run
import plan
import pipeline
import schedule
from test_compute_budget import args, model, PROMPT, SHORT

torch.set_num_threads(1)


class ReachedModelLoad(Exception):
    pass


class Tests(unittest.TestCase):
    def test_all_twelve_planned_commands_reach_model_load(self):
        """Use the real CLI parser, stopping before any checkpoint or data I/O."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, stage in enumerate(s for p in plan.build_specs(root, 100000, 200000)
                                           for s in p['training_stages']):
                command = list(stage['command'][1:])
                command[command.index('--output') + 1] = str(root / f'out{index}')
                command[command.index('--device') + 1] = 'cpu'
                with patch.object(sys, 'argv', command), patch.object(run, 'sha256', return_value='a' * 64), \
                     patch.object(run, 'load_model', side_effect=ReachedModelLoad), \
                     patch.object(torch.cuda, 'is_available', return_value=False):
                    with self.assertRaises(ReachedModelLoad):
                        run.main()

    def test_all_real_model_calls_reconcile_with_compute(self):
        """Observe tensors at model entry independently of ledger instrumentation."""
        tok = run.ScratchTokenizer()
        m = model(tok)
        parameters = sum(p.numel() for p in m.parameters())
        observed = []
        def hook(module, positional, keywords):
            batch, query = positional[0].shape
            keys = keywords.get('cache_start', 0) + query
            forward = 2 * parameters * batch * query + 4 * module.config.n_layers * module.config.d_model * batch * query * keys
            observed.append((forward * (3 if torch.is_grad_enabled() else 1), torch.is_grad_enabled()))
        handle = m.register_forward_pre_hook(hook, with_kwargs=True)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            retained = root / 'retention.jsonl'
            retained.write_text(json.dumps({**SHORT, 'sampling_weight': 1.0}) + '\n')
            options = args(root, command='rl', steps=4, max_flop_proxy=1,
                           retention_data=str(retained), policy_epochs=2, microbatch=3)
            result = run.train_rl(m, tok, [{'prompt': PROMPT, 'sampling_weight': .6},
                                          {'prompt': SHORT['prompt'], 'sampling_weight': .4}], options)
        handle.remove()
        self.assertEqual(sum(x[0] for x in observed), result['actual_flop_proxy'])
        phases = result['compute']['phases']
        self.assertEqual(len(observed), sum(x['forward_calls'] for x in phases.values()))
        self.assertEqual(sum(x[1] for x in observed), sum(x['backward_calls'] for x in phases.values()))
        self.assertEqual(set(phases), {'generation_prefill', 'generation_decode',
                                      'behavior_teacher_audit', 'policy', 'retention'})
        self.assertEqual(result['actual_steps'], 1)
        self.assertLessEqual(result['actual_flop_proxy'], result['worst_case_one_step_flop_proxy'])

    def test_invalid_rl_marker_rejected_before_protected_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'rl' / 'final.pt'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'invented checkpoint')
            (checkpoint.parent / 'completed.json').write_text(json.dumps({
                'stage': 'sft', 'final_sha256': 'a' * 64,
                'budget_reached': False, 'stop_reason': 'step_limit'}))
            release = root / 'release.json'
            release.write_text(json.dumps({'checkpoints': ['b' * 64], 'data_sha256': ['c' * 64],
                'completed_rl_checkpoint': str(checkpoint), 'completed_rl_checkpoint_sha256': 'a' * 64}))
            protected = root / 'textbook_dev.jsonl'
            def checked_hash(path):
                self.assertNotEqual(Path(path), protected, 'Protected hash happened before RL completion validation')
                return 'a' * 64
            with patch.object(run, 'sha256', side_effect=checked_hash):
                with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
                    run.guard_development_release(protected, 'b' * 64, release)

    def test_real_cpu_training_metadata_passes_pipeline_and_release_gate(self):
        """Exercise actual serialized SFT/RL schemas without any benchmark fixture."""
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            tok = run.ScratchTokenizer()
            parent = root / 'parent.pt'
            run.save_checkpoint(parent, model(tok), tok, stage='prepared')
            data = root / 'training.jsonl'
            data.write_text(json.dumps({**SHORT, 'sampling_weight': 1.0}) + '\n')
            trained = {}
            for command in ('sft', 'rl'):
                output = root / command
                argv = ['run.py', command, '--checkpoint', str(parent), '--data', str(data),
                        '--output', str(output), '--mode', 'direct', '--device', 'cpu',
                        '--precision', 'fp32', '--steps', '3', '--max-flop-proxy', '1',
                        '--flop-milestones', '1', '--save-every', '0', '--microbatch', '2',
                        '--batch-size', '2', '--prompts-per-step', '2', '--group-size', '2',
                        '--max-new-tokens', '4', '--decode-batch', '4', '--max-logprob-error', '0.0001',
                        '--allow-unguarded-smoke']
                if command == 'rl':
                    argv += ['--retention-data', str(data), '--retention-batch', '2']
                with patch.object(sys, 'argv', argv), patch.object(torch.cuda, 'is_available', return_value=False):
                    run.main()
                manifest = json.loads((output / 'manifest.json').read_text())
                stage = {'output': command, 'command_name': command, 'target_flop_proxy': 1,
                         'checkpoint': str(parent), 'expected_args': manifest['args'], 'milestones': [1]}
                trained[command] = pipeline.validate_training(stage, root)
                parent = output / 'final.pt'
            release = root / 'release.json'
            release.write_text(json.dumps({'checkpoints': [trained['sft']['checkpoint_sha256'], trained['rl']['checkpoint_sha256']],
                'data_sha256': ['c' * 64], 'completed_rl_checkpoint': trained['rl']['checkpoint'],
                'completed_rl_checkpoint_sha256': trained['rl']['checkpoint_sha256']}))
            protected = root / 'textbook_dev.jsonl'
            original_hash = run.sha256
            calls = []
            def guarded_hash(path):
                if Path(path) == protected:
                    calls.append(path)
                    return 'c' * 64
                return original_hash(path)
            with patch.object(run, 'sha256', side_effect=guarded_hash):
                run.guard_development_release(protected, trained['sft']['checkpoint_sha256'], release)
            self.assertEqual(len(calls), 1)
            self.assertFalse(protected.exists())
            done_path = root / 'rl/completed.json'
            done = json.loads(done_path.read_text())
            done['budget_reached'] = False
            done_path.write_text(json.dumps(done))
            calls.clear()
            with patch.object(run, 'sha256', side_effect=guarded_hash), self.assertRaises(ValueError):
                run.guard_development_release(protected, trained['rl']['checkpoint_sha256'], release)
            self.assertEqual(calls, [])

    def test_scheduler_claim_is_durable_before_any_launch(self):
        """An interrupted first launch must leave state that blocks automatic restart."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'reports').mkdir()
            (root / 'plans').mkdir()
            (root / 'plans/protocol.json').write_text(json.dumps({'arms': ['invented_arm'], 'input_sha256': {}}))
            (root / 'plans/invented_arm.json').write_text('{}')
            def before_launch(*arguments, **keywords):
                marker = root / 'reports/scheduler_state.json'
                self.assertTrue(marker.exists(), 'Launch precedes durable scheduler state')
                self.assertIn('invented_arm', marker.read_text())
                raise ReachedModelLoad
            with patch.object(sys, 'argv', ['schedule.py', '--root', str(root)]), \
                 patch.object(schedule, 'state', return_value={2: {'pids': [], 'memory': 0, 'util': 0}}), \
                 patch.object(schedule, 'remote_state', return_value={}), \
                 patch.object(schedule.subprocess, 'run'), \
                 patch.object(schedule.subprocess, 'Popen', side_effect=before_launch):
                with self.assertRaises(ReachedModelLoad):
                    schedule.main()


if __name__ == '__main__':
    unittest.main()
