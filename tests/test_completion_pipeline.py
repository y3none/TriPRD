"""Single-route process ordering, GPU allocation and restart contract checks."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from tools import run_triprd_completion_pipeline as pipeline


class CompletionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(prefix="completion-pipeline-")
        self.addCleanup(self.folder.cleanup)
        self.root=Path(self.folder.name)
        self.manifest=self.root/"manifest.json"
        self.manifest.write_text('{}',encoding="utf-8")
        self.a=pipeline.parse_args(["--output-root",str(self.root/"out"),"--split-manifest",str(self.manifest)])

    def test_one_route_uses_all_requested_gpus_and_global_batch(self):
        plan=pipeline.build_plan(self.a)
        self.assertEqual(plan['gpus'],['2','3','0','1','6'])
        self.assertEqual(plan['batch_size_global'],8)
        self.assertTrue(plan['data_parallel'])
        self.assertEqual([j['stage'] for j in plan['jobs']],['teacher','source','cod','test'])
        for job in plan['jobs'][:3]:
            self.assertIn('--data-parallel',job['command'])
        source,cod=plan['jobs'][1:3]
        self.assertIn('--teacher-checkpoint',source['command'])
        self.assertNotIn('--source-checkpoint',source['command'])
        self.assertIn('--source-checkpoint',cod['command'])
        self.assertNotIn('--teacher-checkpoint',cod['command'])
        self.assertNotIn('--group',cod['command'])
        self.assertEqual(cod['command'][cod['command'].index('--cod-train-mode')+1],'full')

    def test_development_pipeline_never_schedules_official_test(self):
        self.a.cod_train_mode='split'
        self.assertEqual([j['stage'] for j in pipeline.build_plan(self.a)['jobs']],list(pipeline.STAGES))
        self.a.phase='test'
        with self.assertRaisesRegex(ValueError,'full4040'):
            pipeline.build_plan(self.a)

    def test_single_gpu_does_not_enable_parallel_and_bad_ids_rejected(self):
        self.a.gpus='6'
        plan=pipeline.build_plan(self.a)
        self.assertFalse(plan['data_parallel'])
        self.assertNotIn('--data-parallel',plan['jobs'][0]['command'])
        self.a.gpus='2,2'
        with self.assertRaises(ValueError):
            pipeline.build_plan(self.a)
        self.a.gpus='2,3,0,1,6'
        self.a.batch_size=4
        with self.assertRaisesRegex(ValueError,'batch'):
            pipeline.build_plan(self.a)

    def test_resume_passed_once_to_every_training_stage(self):
        self.a.resume=True
        for job in pipeline.build_plan(self.a)['jobs'][:3]:
            self.assertEqual(job['command'].count('--resume'),1)

    def fixture(self,fail=False):
        plan=pipeline.build_plan(self.a)
        marker=self.root/'ran.jsonl'
        for job in plan['jobs']:
            script=self.root/(job['stage']+'.py')
            code=("import json,os\nfrom pathlib import Path\n"
                  f"with Path({str(marker)!r}).open('a') as f: f.write(json.dumps(["
                  f"{job['stage']!r},os.environ.get('CUDA_VISIBLE_DEVICES')])+ '\\n')\n"
                  f"print({job['stage']!r},flush=True)\n")
            if fail and job['stage']=='source':
                code+='raise SystemExit(7)\n'
            script.write_text(code,encoding='utf-8')
            job['command']=[sys.executable,str(script)]
        return plan,marker

    def test_real_children_order_logs_cuda_visibility_and_configuration_lock(self):
        plan,marker=self.fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.execute_plan(plan,self.a),0)
        rows=[json.loads(line) for line in marker.read_text().splitlines()]
        self.assertEqual([r[0] for r in rows],['teacher','source','cod','test'])
        self.assertEqual({r[1] for r in rows},{'2,3,0,1,6'})
        status=json.loads((self.a.output_root/'pipeline_status.json').read_text())
        self.assertEqual(status['status'],'complete')
        self.assertFalse((self.a.output_root/'completion_pipeline.lock').exists())
        self.a.batch_size=10
        with self.assertRaisesRegex(ValueError,'configuration changed'):
            pipeline.execute_plan(plan,self.a)

    def test_failed_stage_stops_dependents_and_records_exit_code(self):
        plan,marker=self.fixture(fail=True)
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'source failed'):
            pipeline.execute_plan(plan,self.a)
        rows=[json.loads(line) for line in marker.read_text().splitlines()]
        self.assertEqual([r[0] for r in rows],['teacher','source'])
        status=json.loads((self.a.output_root/'pipeline_status.json').read_text())
        self.assertEqual(status['status'],'failed')
        self.assertEqual(status['jobs']['source']['exit_code'],7)
        self.assertFalse((self.a.output_root/'completion_pipeline.lock').exists())


if __name__=='__main__':
    unittest.main()
