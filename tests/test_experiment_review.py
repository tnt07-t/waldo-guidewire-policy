"""Saved-run reader and notebook checks using invented reports only."""
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from experiment_review import default_roots, discover_runs, render_review, prepare_notebook, link_journal, review_saved_experiments


class ReviewTests(unittest.TestCase):
    def make_run(self, root, name='invented', *, model=None, architecture=None):
        directory=Path(root)/name
        directory.mkdir()
        config=dict(optimizer='Adam', lr=.001, weight_decay=0, batch_size=32, split_position=0, seed=0)
        if model is not None: config['model']=model
        if architecture is not None: config['architecture']=architecture
        def scores(ba):
            return dict(loss=.7, accuracy=.6, balanced_accuracy=ba, retract_recall=.5,
                        advance_recall=.7, confusion_matrix=[[1,1],[1,3]],n=6)
        files={
            'run_context.json': dict(config=config, git_commit='invented-sha', git_dirty=False,
                                     train_rows=['NEVER_SHOW_ROW'], metadata_scale_depth_time=['NEVER_SHOW_SCALE']),
            'summary.json': dict(status='completed', best_epoch=1, completed_epochs=2,
                                  updates=2, best_validation=scores(.6), reload_verified=True, encoder_unchanged=True),
            'history.json': [dict(epoch=1, train=scores(.8), validation=scores(.6)),
                             dict(epoch=2, train=scores(.9), validation=scores(.5))]}
        for name,value in files.items():
            (directory/name).write_text(json.dumps(value))
        return directory

    def test_legacy_and_static_identity_selected_final_and_no_private_fields(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_run(root,'late')
            self.make_run(root,'early',architecture='early_goal')
            self.make_run(root,'static',model='static_latest_goal_metadata')
            runs,warnings=discover_runs([root])
            self.assertFalse(warnings)
            self.assertEqual({r['approach'] for r in runs}, {'GRU / late_goal','GRU / early_goal','Static latest + goal + metadata'})
            for run in runs:
                self.assertEqual(run['selected_train']['balanced_accuracy'], .8)
                self.assertEqual(run['final']['validation']['balanced_accuracy'], .5)
            html=render_review(runs)
            self.assertNotIn('NEVER_SHOW',html)
            self.assertNotIn('NEVER_SHOW',json.dumps(runs))
            for svg in re.findall(r'<svg.*?</svg>', html, re.S):
                ET.fromstring(svg)
            self.assertIn('RETRACT recall', html)
            self.assertIn('Trial number (report-file order)', html)
            self.assertIn('Selected epoch 1', html)
            self.assertIn('Selected train',html)

    def test_missing_corrupt_and_partial_runs_stay_visible(self):
        with tempfile.TemporaryDirectory() as root:
            good=self.make_run(root,'incomplete')
            (good/'summary.json').unlink()
            (good/'run_context.json').unlink()
            corrupt=self.make_run(root,'corrupt')
            (corrupt/'summary.json').write_text('{bad json')
            history=json.loads((corrupt/'history.json').read_text())
            history.reverse()
            (corrupt/'history.json').write_text(json.dumps(history))
            runs,_=discover_runs([root])
            self.assertEqual({r['status'] for r in runs}, {'incomplete','report_error'})
            partial=next(r for r in runs if r['status']=='incomplete')
            self.assertEqual(partial['approach'], 'Unknown model (context missing)')
            self.assertFalse(partial['selected_train'])
            self.assertTrue(next(r for r in runs if r['status']=='report_error')['problems'])
            self.assertIn('report_error',render_review(runs))

    def test_probe_reports_have_no_fabricated_epochs_or_seed(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/'probe'; directory.mkdir()
            (directory/'metrics.json').write_text(json.dumps(dict(split_position=0,results={
                'dinov2:latest_goal': dict(train=dict(balanced_accuracy=.8),
                                           validation=dict(balanced_accuracy=.6),selected_C=.1)})))
            (directory/'validation_predictions.private.json').write_text('PRIVATE DO NOT READ')
            runs,_=discover_runs([root])
            self.assertEqual(len(runs),1)
            self.assertEqual(runs[0]['kind'],'probe')
            self.assertNotIn('seed',runs[0]['config'])
            self.assertFalse(runs[0]['history'])
            html=render_review(runs)
            self.assertNotIn('PRIVATE DO NOT READ',html)
            self.assertIn('no epoch history',html)

    def test_escaped_labels_notes_sorted_filter_and_no_writes(self):
        with tempfile.TemporaryDirectory() as root:
            a=self.make_run(root,'a')
            b=self.make_run(root,'b')
            for name in ('summary.json','run_context.json','history.json'):
                os.utime(a/name,(100,100)); os.utime(b/name,(200,200))
            runs,_=discover_runs([root])
            before={p: p.read_bytes() for p in Path(root).rglob('*') if p.is_file()}
            selected=runs[0]['id']
            html=render_review(runs,sort_by='validation_ba',selected_ids=[selected],notes={
                selected:dict(label='<script>alert(1)</script>', decision='<unsafe & text>')})
            self.assertNotIn('<script>',html)
            self.assertIn('&lt;script&gt;',html)
            self.assertIn('&lt;unsafe &amp; text&gt;',html)
            self.assertEqual(html.count('<summary>Trial '),1)
            after={p: p.read_bytes() for p in Path(root).rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            with self.assertRaises(ValueError): render_review(runs,sort_by='unknown')

    def test_missing_roots_limits_and_oversized_symlink_reports(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(default_roots(),[])
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            self.make_run(root,'a'); b=self.make_run(root,'b')
            runs,warnings=discover_runs([root],max_runs=1)
            self.assertEqual(len(runs),1); self.assertTrue(warnings)
            (b/'summary.json').unlink()
            external=Path(outside)/'summary.json'; external.write_text('{"status":"completed"}')
            (b/'summary.json').symlink_to(external)
            runs,_=discover_runs([root])
            self.assertEqual(next(r for r in runs if r['id'].endswith('/b'))['status'],'report_error')
            with patch('experiment_review.MAX_JSON_BYTES',1):
                runs,_=discover_runs([root])
                self.assertTrue(all(r['status']=='report_error' for r in runs))
            runs,warnings=discover_runs([Path(root)/'missing'])
            self.assertFalse(runs); self.assertTrue(warnings)
            self.assertIn('No saved runs found',render_review(runs,warnings))

    def test_private_notebook_creation_is_output_free_and_preserves_existing_notes(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, PSCRATCH=root):
            path=prepare_notebook()
            self.assertEqual(path,Path(root).resolve()/'waldo-review'/'experiment_review.ipynb')
            saved=json.loads(path.read_text())
            source=''.join(''.join(c['source']) for c in saved['cells'])
            self.assertIn('sys.path.insert',source)
            self.assertIn(str(Path(root)/'waldo-gru-training'),source)
            self.assertTrue(all(c.get('outputs',[])==[] for c in saved['cells']))
            path.write_text('PRIVATE USER NOTES')
            self.assertEqual(prepare_notebook(),path)
            self.assertEqual(path.read_text(),'PRIVATE USER NOTES')

    def fake_journal(self, hint=None):
        return dict(format_version=1,takeaway='Invented history only',trials=[dict(
            id='fake-trial',title='Invented baseline',date='2000-01-01',
            best_epoch=1,completed_epochs=2,updates=2,artifact_hint=hint,
            protocol=dict(optimizer='Adam',seed=0,split_position=0),
            validation=dict(balanced_accuracy=.6),selected_train=dict(balanced_accuracy=.8),
            question='Invented question',change='Invented change',observation='Invented finding',
            decision='Invented decision',provenance='Synthetic fixture',source='invented.md')],
            other_work=[],next_options=[],probe_baseline={})

    def test_journal_links_by_fingerprint_without_overwriting_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_run(root)
            runs,_=discover_runs([root])
            before=json.dumps(runs,sort_keys=True)
            linked=link_journal(self.fake_journal(),runs)
            self.assertEqual(linked[0]['artifact_id'],runs[0]['id'])
            self.assertEqual(json.dumps(runs,sort_keys=True),before)
            with patch('experiment_review.load_journal',return_value=self.fake_journal()):
                html=review_saved_experiments([root])
            self.assertIn('Invented decision',html)
            self.assertIn('Saved report linked automatically',html)
            self.assertIn('Invented baseline — train',html)
            self.assertIn('Final evaluated validation',html)

    def test_duplicate_and_conflicting_report_links_do_not_choose_silently(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_run(root,'a'); b=self.make_run(root,'b')
            runs,_=discover_runs([root])
            linked=link_journal(self.fake_journal(),runs)
            self.assertIsNone(linked[0]['artifact_id'])
            self.assertEqual(len(linked[0]['candidates']),2)
            bad=json.loads((b/'summary.json').read_text())
            bad['best_validation']['balanced_accuracy']=.1
            (b/'summary.json').write_text(json.dumps(bad))
            runs,_=discover_runs([root])
            journal=self.fake_journal(next(r['id'] for r in runs if r['id'].endswith('/b')))
            # A matching unknown-name report can still be linked, while the wrong
            # documented-path evidence remains an explicit conflict notice.
            linked=link_journal(journal,[r for r in runs if r['id'].endswith('/b')])
            self.assertIsNone(linked[0]['artifact_id'])
            self.assertTrue(linked[0]['conflicts'])

    def test_history_remains_useful_without_saved_artifacts_or_code_edits(self):
        with tempfile.TemporaryDirectory() as root,patch('experiment_review.load_journal',return_value=self.fake_journal()):
            html=review_saved_experiments([root])
            self.assertIn('Invented baseline',html)
            self.assertIn('60.00%',html)
            self.assertIn('Recorded result retained; saved report not located',html)
            self.assertIn('No saved epoch history located',html)
            self.assertNotIn('stroke-dasharray="3 4"',html)
            self.assertNotIn('Set ROOTS',html)

    def test_recent_extra_runs_get_curves_without_manual_selection(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_run(root,'known')
            extra=self.make_run(root,'new-run',model='static_grid_goal_metadata')
            summary=json.loads((extra/'summary.json').read_text())
            summary['best_validation']['balanced_accuracy']=.7
            (extra/'summary.json').write_text(json.dumps(summary))
            with patch('experiment_review.load_journal',return_value=self.fake_journal()):
                html=review_saved_experiments([root])
            self.assertIn('new-run',html)
            self.assertIn('Static spatial grid + goal + metadata',html)
            self.assertEqual(html.count('<summary>Trial '),2)
            self.assertIn('70.00%',html)

    def test_notebook_is_output_free_and_cells_execute_on_synthetic_reports(self):
        notebook=json.loads((Path(__file__).resolve().parents[1]/'experiment_review.ipynb').read_text())
        code=[c for c in notebook['cells'] if c['cell_type']=='code']
        self.assertTrue(all(c['outputs']==[] and c['execution_count'] is None for c in code))
        outputs=[]
        ipython=ModuleType('IPython'); display=ModuleType('IPython.display')
        display.HTML=lambda value:value
        display.display=outputs.append
        with tempfile.TemporaryDirectory() as root:
            self.make_run(Path(root),'waldo-gru-training')
            with patch.dict(os.environ,PSCRATCH=root),patch.dict(sys.modules,{'IPython':ipython,'IPython.display':display}), patch('experiment_review.load_journal',return_value=self.fake_journal()):
                namespace={}
                for cell in code:
                    exec(compile(''.join(cell['source']),'<review-notebook>','exec'),namespace)
        self.assertEqual(len(outputs),1)
        self.assertIn('GRU / late_goal',outputs[0])


if __name__=='__main__':
    unittest.main()
