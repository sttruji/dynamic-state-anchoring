"""Offline four-policy execution and notebook boundaries for the next experiment."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

import torch
from dsa.data.pilot_prepare import prepare_pilot
from scripts.diagnose_dsa_pilot import diagnose
from scripts.ensure_pilot_arm import ensure_arm
from scripts.evaluate_dsa import evaluate_run
from scripts.package_pilot_review import package_review
from scripts.pilot_explicit_controls import explicit_controls
from scripts.screen_recurrence import screen

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
CONDITIONS = {mode:dict(recurrence_mode="flt_memory", write_mode=mode)
              for mode in ("static", "every_loop", "periodic", "learned")}


class DataScaleNotebookTests(unittest.TestCase):
    def test_configuration_and_executable_cells_keep_compute_bounded(self):
        n = json.loads((ROOT/'notebooks/DSA_FLT_DataScale.ipynb').read_text())
        sources = {c['id']:''.join(c['source']) for c in n['cells']}
        self.assertEqual(n['cells'][0]['id'], 'clone')
        self.assertEqual(len(sources), len(n['cells']))
        for c in n['cells']:
            if c['cell_type'] == 'code':
                compile(sources[c['id']], c['id'], 'exec')
                self.assertIsNone(c['execution_count']); self.assertEqual(c['outputs'], [])
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            namespace = dict(Path=Path, json=json, hashlib=hashlib, os=os, sys=sys,
                             PROJECT_DIR=Path(temp), REPO_DIR=ROOT, REPO_COMMIT='fixture', PYTHON=sys.executable)
            exec(compile(sources['config'], 'config', 'exec'), namespace)
            self.assertEqual(namespace['CONDITIONS'], CONDITIONS)
            self.assertEqual(namespace['TRAIN_LIMIT'], 512)
            self.assertEqual(namespace['STEPS'], 1024)
            self.assertEqual(namespace['VALID_LIMIT'], 0)
            self.assertEqual(namespace['TRAIN_DIAGNOSTIC_LIMIT'], 64)
            self.assertEqual(namespace['GRADIENT_LIMIT'], 2)
            self.assertEqual(namespace['DEPTH_GRID'], [1,4,8])
            self.assertEqual(namespace['EXPERIMENT_KIND'], 'data_scale')
            run_name = namespace['RUN_NAME']
            # Requesting restart changes execution policy, not experiment identity.
            exec(compile(sources['config'].replace('RUN_MODE = "auto_pilot"', 'RUN_MODE = "restart_all"'), 'restart-config', 'exec'), namespace)
            self.assertEqual(namespace['RUN_NAME'], run_name)
        self.assertIn('"--training-limit", TRAIN_DIAGNOSTIC_LIMIT', sources['pilot-diagnostics'])
        self.assertIn('RUN_TEST = False', sources['final-evaluation'])
        ids = list(sources)
        self.assertLess(ids.index('training'), ids.index('evaluation'))
        self.assertLess(ids.index('pilot-controls'), ids.index('pilot-export'))


class FourPolicyDataScaleTests(unittest.TestCase):
    def test_complete_workflow_caps_diagnostics_reuses_backups_and_exports_all_pairs(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            project = Path(temp); source = project/'source'; source.mkdir()
            for split in ('train','valid','test'):
                records = [dict(question=f'{split} {i}: twice the sum of {i} and 3?',
                                steps=[f'<<{i}+3={i+3}>>',f'<<{i+3}*2={2*(i+3)}>>'],
                                answer=str(2*(i+3))) for i in range(1,4)]
                (source/f'gsm_{split}.json').write_text(json.dumps(records))
            prepared = project/'prepared/cache'
            prepare_pilot(storage_dir=None, source_dir=source, output_dir=prepared, loops=2, tiny=True, train_limit=0)
            runs = project/'vm'; backups = project/'backups'; legacy = project/'runs/fixture'
            review = project/'reports/fixture/review'
            initial = screen(prepared, review/'initialization-screen', device='cpu',
                             conditions=CONDITIONS, depths=(1,2))
            self.assertTrue(initial['finite'])
            sample_ids, orders = [], []
            for arm,condition in CONDITIONS.items():
                folder = review/arm
                result = ensure_arm(prepared, runs/arm, legacy/arm, backups,
                                    folder/'checkpoint-status.json', arm=arm, steps=3, device='cpu',
                                    expected_initial_sha=initial['initial_state_sha256'], **condition)
                self.assertEqual(result['action'], 'restarted')
                self.assertEqual(result['backup_status'], 'verified_mounted_copy')
                journal = [json.loads(line) for line in (runs/arm/'metrics.jsonl').read_text().splitlines()]
                orders.append([r['sample_ids'] for r in journal])
                report = diagnose(prepared, runs/arm, folder/'diagnostics', device='cpu',
                                  training_limit=2, gradient_limit=1, probe_limit=1,
                                  depth_grid=(1,2), max_new_tokens=2, comprehensive=True)
                self.assertEqual(report['full_training_examples'], 3)
                self.assertEqual(report['training_examples'], 2)
                self.assertFalse(report['all_training_examples_profiled'])
                self.assertEqual(len(report['forward_reports']), 2)
                self.assertEqual(report['prediction_reports']['normal_training_set']['total'], 2)
                self.assertEqual(len(report['gradient_reports']), 3)
                sample_ids.append(report['identity']['selected_training_sample_ids'])
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    diagnose(prepared, runs/arm, folder/'diagnostics', device='cpu', training_limit=1,
                             gradient_limit=1, probe_limit=1, depth_grid=(1,2), max_new_tokens=2, comprehensive=True)
                evaluate_run(prepared, runs/arm, folder/'validation-step3-k2-n0-g2',
                             device='cpu', limit=0, max_new_tokens=2)
                explicit_controls(prepared, folder/'explicit-control', run_dir=runs/arm,
                                  device='cpu', limit=1, max_new_tokens=2)
            self.assertTrue(all(ids == sample_ids[0] for ids in sample_ids))
            self.assertTrue(all(order == orders[0] for order in orders))
            # A new VM path must restore the completed backup without retraining.
            restored = ensure_arm(prepared, project/'other-vm/learned', legacy/'learned', backups,
                                  review/'restored-status.json', arm='learned', steps=3, device='cpu',
                                  expected_initial_sha=initial['initial_state_sha256'], **CONDITIONS['learned'])
            self.assertEqual(restored['action'], 'reused')
            explicit_controls(prepared, review/'original-explicit-control', device='cpu', limit=1, max_new_tokens=2)
            (review/'protocol.json').write_text(json.dumps(dict(EXPERIMENT_KIND='data_scale', CONDITIONS=CONDITIONS)))
            (review/'session_status.json').write_text('{"offline-workflow":{"status":"passed"}}')
            manifest = package_review(project, 'fixture', 'review', prepared, arms=list(CONDITIONS), run_root=runs)
            self.assertTrue(manifest['evidence_complete'])
            with ZipFile(project/'bundles/fixture-review.zip') as z:
                summary = json.loads(z.read('review/scale_readiness.json'))
                self.assertEqual(len(summary['write_policy_validation_comparisons']), 6)
                self.assertFalse(any('80%' in x for x in summary['blocking']))
                self.assertIn('source/notebooks/DSA_FLT_DataScale.ipynb', z.namelist())
                self.assertNotIn('prepared/test.jsonl', z.namelist())
                for arm in CONDITIONS:
                    self.assertEqual(summary['arms'][arm]['validation']['total'], 3)
                    self.assertEqual(summary['arms'][arm]['training_diagnostic_scope']['profiled_records'], 2)
                    self.assertEqual(summary['arms'][arm]['training_diagnostic_scope']['full_training_records'], 3)
                    self.assertIn('auc', summary['arms'][arm]['controller_event_discrimination'])
                for name,entry in manifest['included'].items():
                    self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(), entry['sha256'])


if __name__ == '__main__':
    unittest.main()
