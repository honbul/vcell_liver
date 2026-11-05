"""Exercise stage selection and the frozen evaluator with artificial results only."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from artifact_io import digest, write
from common_effects import summarize
import common_pipeline
from common_selection import (NEURAL, PROTOCOL, ROOT, choose_confirmation, declared_screen,
                              lock_test, read, select_model)


class CommonConfirmationTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name)
        self.runs = self.path/'runs'
        self.cfg = read(PROTOCOL)
        self.dataset_sha = 'fixture_only'

    def validation_run(self, job, delta):
        path = self.runs/job['phase']/job['run_id']
        path.mkdir(parents=True)
        (path/'best_model.pt').write_bytes(b'fixture')
        fields = {k:job[k] for k in ('model','mode','data_mode','seed','lr')}
        write(path/'settings.json', dict(contract={
            **fields, 'dataset_sha256':self.dataset_sha, 'protocol_sha256':digest(NEURAL),
            'adapter_sha256':{'common_train.py':digest(ROOT/'src/common_train.py')}
        }))
        write(path/'best_validation.json', dict(role='val',n_available=34,n_scored=34,
                                               metrics={'delta_rmse':delta}))
        write(path/'result.json', dict(
            **job, status='validation_complete', dataset_sha256=self.dataset_sha,
            best_model_sha256=digest(path/'best_model.pt'), validation={'delta_rmse':delta},
            model_init={'n_trainable':1}, gpu='fixture', epoch_seconds=1.0))
        return path

    def select(self, model, data):
        jobs = declared_screen(self.cfg)
        scores = [1.0, 1.2, model, model+.02, model+.04, .5]
        for job,score in zip(jobs,scores):
            self.validation_run(job,score)
        pick = select_model(self.cfg,self.runs,self.dataset_sha)
        for job,score in zip(pick['jobs'],data):
            self.validation_run(job,score)
        return pick, choose_confirmation(self.cfg,pick,self.runs,self.dataset_sha)

    def test_positive_factorial_and_checkpoint_binding(self):
        pick, plan = self.select(.9,(.8,.95))
        self.assertEqual(pick['fixed_model']['job']['mode'],'M1')
        self.assertEqual(set(plan['arms']),{'D0M0','D0M*','D*M0','D*M*'})
        self.assertEqual(plan['primary_arm'],'D*M*')
        self.assertEqual(len(plan['repeats']),12)
        self.assertEqual(sum(r['reused_validation'] for r in plan['repeats']),3)
        for row in plan['repeats']:
            if not row['reused_validation']:
                self.validation_run(row['job'],.9)
        baseline = self.path/'simple.json'
        write(baseline,dict(role='val',dataset_sha256=self.dataset_sha,
                            run_code_sha256=digest(ROOT/'src/common_baselines.py'),selected_alpha=.1))
        lock = lock_test(plan,self.runs,baseline)
        self.assertEqual(len(lock['rows']),12)
        job = lock['rows'][0]['job']
        (self.runs/job['phase']/job['run_id']/'best_model.pt').write_bytes(b'other')
        with self.assertRaisesRegex(ValueError,'Checkpoint hash mismatch'):
            lock_test(plan,self.runs,baseline)

    def test_negative_screens_still_repeat_one_alternative(self):
        pick, plan = self.select(1.05,(1.01,1.04))
        self.assertEqual(pick['fixed_model']['job']['mode'],'M0')
        self.assertEqual(set(plan['arms']),{'D0M0','D*M0'})
        self.assertEqual(len(plan['repeats']),6)
        self.assertEqual(plan['primary_arm'],'D*M0')

    def test_data_transition_submits_only_two_ablation_jobs_and_gates_confirmation(self):
        work, dataset = self.path/'study', self.path/'data'
        work.mkdir()
        dataset.mkdir()
        write(dataset/'receipt.json',{'split_seed':self.cfg['initial_split']})
        self.dataset_sha = digest(dataset/'receipt.json')
        self.runs = self.path/'assets/runs'
        for j,s in zip(declared_screen(self.cfg),(1.0,1.2,.91,.92,.93,.65)):
            self.validation_run(j,s)
        pick = select_model(self.cfg,self.runs,self.dataset_sha)
        write(work/'model_selection.json',pick)
        write(work/'paper_status.json',dict(
            R1={'status':'modified_reproduction_executed','evidence':{'fixture':True}},
            R2={'status':'modified_reproduction_executed','evidence':{'fixture':True}}))
        requests = []

        def fake_submit(key,args,journal):
            requests.append((key,args))
            return str(900+len(requests))

        with patch.object(common_pipeline,'ROOT',self.path), patch.object(common_pipeline,'submit',fake_submit):
            common_pipeline.transition(SimpleNamespace(stage='data',primary_workspace=work,
                                                       primary_dataset=dataset,split=26))
        self.assertEqual([r[0] for r in requests[:2]],[j['run_id'] for j in pick['jobs']])
        self.assertEqual(requests[2][0],'pipeline-confirmation-s26')
        self.assertEqual(requests[2][1][0],'--dependency=afterok:901:902')
        self.assertTrue(all('M1' in j[1] for j in requests[:2]))
        self.assertEqual(len(requests),3)

    def test_reserve_summary_pairs_conditions_despite_different_row_order(self):
        freeze = self.path/'frozen.json'
        scores = self.path/'scores'
        scores.mkdir()
        rows = [dict(arm=arm,job={'run_id':run,'seed':0},checkpoint_sha256=run)
                for arm,run in [('D0M0','base'),('D0M*','candidate')]]
        write(freeze,dict(stage='frozen_test',reserve=True,protocol_sha256=digest(PROTOCOL),
                          primary_arm='D0M*',dataset_sha256=self.dataset_sha,rows=rows))
        for row in rows:
            run = row['job']['run_id']
            values = [('B',.8),('A',1.0)] if run=='base' else [('A',.6),('B',.7)]
            artifact = scores/f'{run}.predictions.npz'
            artifact.write_bytes(run.encode())
            write(scores/f'{run}.json',dict(
                prediction_sha256=digest(artifact), freeze_sha256=digest(freeze), dataset_sha256=self.dataset_sha, run_id=run,
                checkpoint_sha256=run, seed=0, arm=row['arm'],
                metrics={'role':'test','n_available':2,
                         'metrics':{'delta_rmse':sum(v for _,v in values)/2},
                         'per_condition':[dict(condition=c,delta_rmse=v) for c,v in values]}))
        result = summarize(freeze,scores)
        self.assertEqual(result['conditions'],['A','B'])
        self.assertAlmostEqual(result['effects']['D0M*']['mean_gain'],.25)
        self.assertAlmostEqual(result['effects']['D0M*']['condition_gains'][0],.4)
        self.assertEqual(result['followup_reserve_splits'],[])
        item = scores/'candidate.json'
        scored = read(item)
        scored['checkpoint_sha256'] = 'modified'
        write(item,scored)
        with self.assertRaisesRegex(ValueError,'frozen run'):
            summarize(freeze,scores)


if __name__ == '__main__':
    unittest.main()
