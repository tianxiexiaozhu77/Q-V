"""CPU regression tests for complete-scheme eligibility and restart deduplication."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_current_queue as report


def fixture(phase, keys, names):
    rows, summaries = [], {}
    for name, gain in names.items():
        candidate_rows = []
        for model, dataset in keys:
            for metric in report.METRICS[report.TASKS[model, dataset]]:
                candidate_rows.append({'phase': phase, 'candidate': name, 'model': model,
                    'dataset': dataset, 'task': report.TASKS[model, dataset], 'metric': metric,
                    'test_images': 100, 'support': 10, 'original_percent': 80.,
                    'previous_eta03_percent': 81., 'candidate_percent': 80.+gain, 'gain_pp': gain})
        rows.extend(candidate_rows)
        summaries[name] = {'complete_all_tasks': len(keys) == 15, 'minimum_gain_pp': gain,
                           'mean_gain_pp': gain}
    return {'complete_tasks': len(keys), 'shared_fitter_sha256': 'samealgorithm',
            'failed_or_interrupted': [], 'candidates': summaries}, rows


class CurrentQueueTests(unittest.TestCase):
    def test_pair_requires_chestxdet_classification_and_grounding(self):
        _, rows = fixture('phase12', [('alpharad', 'chestxdet10')], {'one': 5.})
        counts = report.complete_pair_counts(rows)
        self.assertEqual(counts['complete_pairs'], 0)
        _, extra = fixture('phase12', [('alpharad', 'chestxdet10_gt')], {'one': -1.})
        counts = report.complete_pair_counts(rows+extra)
        self.assertEqual(counts['complete_pairs'], 1)
        self.assertEqual(counts['all_metrics_positive_pairs'], 0)

    def test_incomplete_policy_never_outranks_full_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def summarize(path):
                if path.name == 'phase6':
                    return fixture(path.name, report.TASK_ORDER, {'original': 0., 'tta_center_refined': 1.2})
                return fixture(path.name, report.TASK_ORDER[:5], {'original': 0., 'huge_partial': 20.})
            with patch.object(report, 'summarize_phase', side_effect=summarize):
                value = report.build_report(root)
            self.assertEqual(value['current_best_complete_unified']['name'], 'tta_center_refined')
            self.assertEqual(len(value['task_coverage']), 15)
            for phase in value['phases'][1:]:
                for candidate in phase['candidates']:
                    self.assertIsNone(candidate['full_scheme_rank'])
                    self.assertFalse(candidate['eligible_for_full_ranking'])
            text = report.render_text(value)
            self.assertIn('不完整候选不参加完整方案排名', text)
            self.assertIn('未完成', text)

    def test_deadline_followup_replaces_old_partial_and_reads_new_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for phase in ('phase12', 'phase12_no_deadline', 'phase14', 'phase14_no_deadline'):
                (root/'runs'/phase).mkdir(parents=True)
            for name, status in (('controller_v2', 'stopped'), ('controller_no_deadline', 'running')):
                (root/name).mkdir()
                (root/name/'state.json').write_text(json.dumps({'status': status, 'deadline': None,
                    'jobs': {'newphase12': {'status': 'running', 'job': {'output': str(root/'runs/phase12_no_deadline/radzero_SIIM')}}}}))
            called = []
            def summarize(path):
                called.append(path.name)
                names = {'original': 0., 'tta_center_refined': 1.2} if path.name == 'phase6' else {'original': 0., 'fixed_policy': 2.}
                keys = report.TASK_ORDER if path.name != 'phase17b' else report.TASK_ORDER[:5]
                return fixture(path.name, keys, names)
            with patch.object(report, 'summarize_phase', side_effect=summarize):
                value = report.build_report(root)
            self.assertEqual(called, ['phase6', 'phase12_no_deadline', 'phase14_no_deadline', 'phase17b'])
            self.assertEqual(value['controller']['status'], 'running')
            self.assertTrue(value['controller']['no_deadline_controller'])
            self.assertEqual(value['controller']['unfinished_jobs'], 1)
            self.assertEqual(value['phases'][1]['queue_status_counts'], {'running': 1})
            self.assertEqual(value['phases'][1]['physical_phase'], 'phase12_no_deadline')
            # Exact tie follows unified_result.py: lexicographically last phase.
            self.assertEqual(value['current_best_complete_unified']['phase'], 'phase14')
            self.assertEqual(len([r for r in value['metric_rows'] if r['phase'] == 'phase12' and r['candidate'] == 'fixed_policy']), 27)

    def test_atomic_publish_preserves_exact_newlines_and_unicode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'CURRENT_QUEUE_RESULTS.txt'
            report.publish(path, '第一行\n第二行\n')
            self.assertEqual(path.read_text(), '第一行\n第二行\n')
            report.publish(path, '更新后的汇总\n')
            self.assertEqual(path.read_text(), '更新后的汇总\n')
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ['CURRENT_QUEUE_RESULTS.txt'])

    def test_stopped_old_controller_is_not_finite_queue_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'controller_v2').mkdir()
            (root/'controller_v2/state.json').write_text(json.dumps({'status': 'stopped',
                'jobs': {'old': {'status': 'not_started'}}}))
            def summarize(path):
                return fixture(path.name, report.TASK_ORDER[:5], {'original': 0., 'one': 1.})
            with patch.object(report, 'summarize_phase', side_effect=summarize):
                value = report.build_report(root)
            self.assertFalse(value['current_queue_finished'])
            self.assertTrue(value['controller']['queue_transition_pending'])
            self.assertIn('旧控制器停止不表示本次任务完成', report.render_text(value))

    def test_finite_queue_requires_all_accepted_and_terminal_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'controller_no_deadline').mkdir()
            (root/'queue_no_deadline.json').write_text(json.dumps({'jobs': [{'id': 'a'}, {'id': 'b'}]}))
            def summarize(path):
                return fixture(path.name, report.TASK_ORDER, {'original': 0., 'tta_center_refined': 1.})
            state = root/'controller_no_deadline/state.json'
            for jobs, status, expected in (({'a': {'status': 'complete'}}, 'stopped', False),
                ({'a': {'status': 'complete'}, 'b': {'status': 'complete'}}, 'running', False),
                ({'a': {'status': 'complete'}, 'b': {'status': 'failed'}}, 'stopped', True)):
                state.write_text(json.dumps({'status': status, 'jobs': jobs}))
                with patch.object(report, 'summarize_phase', side_effect=summarize):
                    self.assertEqual(report.build_report(root)['current_queue_finished'], expected)

    def test_ranking_matches_unified_pair_median_not_mean(self):
        def policy(minima, mean):
            return {'eligible_for_full_ranking': True, 'all_metrics_positive_pairs': 14,
                'all_metrics_ge_1pp_pairs': 0, 'minimum_gain_pp': min(minima), 'mean_gain_pp': mean,
                'target_met': False, 'per_complete_pair_min_gain_pp': {str(i): v for i, v in enumerate(minima)}}
        a = policy([.1]+[.8]*13, .8)
        b = policy([.1]+[.2]*12+[100.], 100.)
        self.assertGreater(report.full_rank_key(a), report.full_rank_key(b))
        self.assertEqual(report.full_rank_key(a), (True, 14, 0, .1, .8))


if __name__ == '__main__':
    unittest.main()
