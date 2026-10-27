"""The v5 decision uses complete matched reports and never cherry-picks metrics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools import summarize_triprd_staged as summary

ROOT = Path(__file__).resolve().parents[1]
H = 'a' * 64
D = 'b' * 64


def training(count=2):
    return dict(steps=1, optimizer_steps=1, amp_skipped_steps=0, primary_images=count,
                source_images=0, seconds=1., peak_cuda_allocated_bytes=0)


def metrics(sm, fw=.8, em=.9, mae=.02):
    values = dict(S_alpha=sm, F_beta_w=fw, E_phi=em, MAE=mae)
    return dict(macro=values.copy(), num_samples=2,
                sources={name: dict(values, num_samples=1) for name in ('COD10K', 'CAMO')})


class StagedSummaryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='.staged-summary-', dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        common = {k: H for k in summary.MATCHED}
        common.update(source_sha256={'fixture.py': H}, model_variant='dinov3_vitb16', pretrained=True, epochs=100, source_epochs=20,
            val_metrics_every=5, train_seed=123, split_seed=456, prediction_protocol='prediction-v1',
            cod_optimizer_reset=True, cod_training_rng_reset=True)
        self.reports = {}
        for group in summary.GROUPS:
            p = dict(common, cod_train=['ct1', 'ct2'], cod_val=['COD10K/cv1', 'CAMO/cv2'],
                vdt_train=[], vdt_val=[], source_provenance=None,
                stage_exposure=dict(optimization=['COD_train'], selection=['COD_val'], teacher_in_stage=False, source_batches_in_cod=0))
            rows = []
            for epoch in range(1, 101):
                row = dict(epoch=epoch, train=training())
                if epoch % 5 == 0:
                    row['validation'] = metrics(.8 + epoch / 1000 + (.01 if group == 'S3' else 0),
                        fw=.82 if group == 'S3' else .8, em=.92 if group == 'S3' else .9, mae=.018 if group == 'S3' else .02)
                rows.append(row)
            r = dict(method=summary.METHOD, stage='cod', group=group, status='complete', formal=False,
                test_status='skipped', test={}, protocol=p, config=dict(common, cod_train_mode='split'),
                sample_counts=dict(cod_train=2, cod_val=2, vdt_train=0, vdt_val=0),
                generic_initial_state_sha256=H, initial_state_sha256=H, cod_decoder_before_sha256=D,
                cod_decoder_after_sha256=D, checkpoint_sha256=H, source_provenance=None,
                stage_exposure=p['stage_exposure'], prediction_protocol='prediction-v1',
                completed_epochs=100, best_epoch=100, best_score=rows[-1]['validation']['macro']['S_alpha'],
                selection_metric='macro_S_alpha', history=rows)
            self.reports['screen/' + group] = r
            if group != 'S0':
                sp = dict(common, cod_train=[], cod_val=[], vdt_train=['vt1', 'vt2'], vdt_val=['vv1', 'vv2'],
                    source_provenance=None, teacher_sha256=H if group in ('S2', 'S3') else None,
                    stage_exposure=dict(optimization=['VDT_train'], selection=['VDT_val'], teacher_in_stage=group in ('S2', 'S3'), source_batches_in_cod=0))
                sr = dict(r, stage='source', protocol=sp, source_provenance=None, stage_exposure=sp['stage_exposure'],
                    sample_counts=dict(cod_train=0, cod_val=0, vdt_train=2, vdt_val=2),
                    selection_metric='source_validation_loss', completed_epochs=20, best_epoch=20, best_score=-.1,
                    history=[dict(epoch=e, train=dict(training(), source_images=2), validation=dict(loss=.1 + (20-e)/100, num_samples=2, objective='supervised_gt_only')) for e in range(1,21)])
                sr['contract_sha256'] = summary.digest(sp)
                prov = dict(checkpoint_sha256=H, contract_sha256=sr['contract_sha256'], group=group,
                    epoch=20, completed_epochs=20, contract=sp, generic_initial_state_sha256=H,
                    initial_state_sha256=H, transfer_scope='shared_only_exclude_decoders', completed_checkpoint_sha256=D)
                r['source_provenance'] = p['source_provenance'] = prov
                self.reports['source/' + group] = sr
        for mode in ('rgb', 'multimodal'):
            self.reports['teachers/' + mode] = dict(method='triprd_task_v4', stage='teacher', teacher_mode=mode,
                status='complete', checkpoint_sha256=H, test_status='skipped', test={},
                protocol=dict(vdt_train=['vt1','vt2'], vdt_val=['vv1','vv2']),
                sample_counts=dict(cod_train=0, cod_val=0, vdt_train=2, vdt_val=2),
                config=dict(teacher_epochs=100), completed_epochs=100,
                history=[dict(epoch=e, train=training()) for e in range(1,101)])
        self.save()

    def save(self):
        for group in summary.GROUPS[1:]:
            source = self.reports['source/' + group]
            source['contract_sha256'] = summary.digest(source['protocol'])
            prov = self.reports['screen/' + group]['source_provenance']
            prov['contract_sha256'] = source['contract_sha256']
        for name, report in self.reports.items():
            report['contract_sha256'] = summary.digest(report['protocol'])
            path = self.root / name / 'comparison.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report), encoding='utf-8')

    def test_benefit_requires_complete_paired_reports_and_all_twenty_points(self):
        result = summary.summarize(self.root)
        self.assertEqual(result['errors'], [])
        self.assertTrue(result['report_consistency_verified'])
        self.assertEqual(result['verdict'], 'provisional_suggests_benefit')
        self.assertEqual(len(result['same_epoch']), 20)
        self.assertEqual(result['same_epoch_wins']['S3-S0']['wins'], 20)
        self.assertEqual(result['source']['S3']['cost']['epochs'], 20)
        self.assertEqual(result['teachers']['multimodal']['cost']['epochs'], 100)
        self.assertIn('not statistical significance', result['rule'])

    def test_never_combine_different_epochs_best_metrics(self):
        r = self.reports['screen/S3']
        r['history'][94]['validation'] = metrics(.90, fw=.99, em=.99, mae=.001)
        r['history'][99]['validation'] = metrics(.91, fw=.7, em=.92, mae=.018)
        self.save()
        result = summary.summarize(self.root)
        self.assertEqual(result['verdict'], 'not_supported')
        self.assertEqual(result['screen']['S3']['selected_epoch'], 100)
        self.assertEqual(result['screen']['S3']['selected_metrics']['F_beta_w'], .7)

    def test_exactly_half_same_epoch_wins_is_not_majority(self):
        r = self.reports['screen/S3']
        for row in r['history']:
            if 'validation' in row and row['epoch'] <= 50:
                row['validation'] = metrics(.7, fw=.82, em=.92, mae=.018)
        self.save()
        result = summary.summarize(self.root)
        self.assertEqual(result['verdict'], 'not_supported')
        self.assertEqual(result['same_epoch_wins']['S3-S0']['wins'], 10)

    def test_missing_or_mismatched_metadata_never_verifies(self):
        good = copy.deepcopy(self.reports)
        for mutate in (lambda r: r.pop('generic_initial_state_sha256'),
                       lambda r: r.update(cod_decoder_after_sha256='c'*64),
                       lambda r: r['config'].update(train_seed=999),
                       lambda r: r.update(best_epoch=95),
                       lambda r: r['history'][0]['train'].update(source_images=1)):
            with self.subTest(mutation=mutate):
                self.reports = copy.deepcopy(good)
                mutate(self.reports['screen/S3'])
                self.save()
                result = summary.summarize(self.root)
                self.assertEqual(result['verdict'], 'incomplete')
                self.assertFalse(result['report_consistency_verified'])

    def test_missing_source_and_teacher_reports_are_not_inferred(self):
        for name in ('source/S1', 'teachers/rgb'):
            path = self.root / name / 'comparison.json'
            data = path.read_bytes()
            path.unlink()
            result = summary.summarize(self.root)
            self.assertEqual(result['verdict'], 'incomplete')
            path.write_bytes(data)

    def test_macro_and_grid_corruption_are_rejected(self):
        self.reports['screen/S3']['history'][4]['validation']['macro']['F_beta_w'] = .99
        self.save()
        self.assertEqual(summary.summarize(self.root)['verdict'], 'incomplete')
        self.reports['screen/S3']['history'][4].pop('validation')
        self.save()
        self.assertEqual(summary.summarize(self.root)['verdict'], 'incomplete')

    def test_cpu_tiny_resource_nulls_are_explicit_smoke_evidence(self):
        for name, report in self.reports.items():
            for row in report['history']:
                row['train']['peak_cuda_allocated_bytes'] = None
            if name.startswith('teachers/'):
                continue
            for container in (report['protocol'], report['config']):
                container.update(model_variant='tiny', pretrained=False, generic_weights_sha256=None,
                    generic_config_sha256=None, dino_source_sha256=None)
        self.save()
        result = summary.summarize(self.root)
        self.assertEqual(result['errors'], [])
        self.assertTrue(result['report_consistency_verified'])
        self.assertEqual(result['evidence_scope'], 'smoke_only')
        self.assertEqual(result['verdict'], 'not_supported')
        self.assertIn('smoke_only', summary.markdown(result))
        self.assertEqual(result['screen']['S0']['cost']['peak_cuda_allocated_bytes'], 0)

    def test_per_source_counts_are_bound_to_validation_members(self):
        for source_name, count in [('COD10K', 0), ('CAMO', 2)]:
            self.reports['screen/S3']['history'][4]['validation']['sources'][source_name]['num_samples'] = count
        self.save()
        result = summary.summarize(self.root)
        self.assertEqual(result['verdict'], 'incomplete')
        self.assertTrue(any('member identities' in e for e in result['errors']))

    def test_changed_actual_checkpoint_is_detected(self):
        (self.root / 'screen/S3/best.pt').write_bytes(b'changed weights')
        result = summary.summarize(self.root)
        self.assertEqual(result['verdict'], 'incomplete')
        self.assertTrue(any('actual checkpoint hash' in e for e in result['errors']))

    def test_negative_and_incomplete_cli_reports_return_zero(self):
        self.reports['screen/S3']['history'][-1]['validation'] = metrics(.91, fw=.7)
        self.save()
        self.assertEqual(summary.main(['--output-root', str(self.root)]), 0)
        self.assertTrue((self.root / 'analysis/staged_summary.md').is_file())
        (self.root / 'screen/S0/comparison.json').unlink()
        self.assertEqual(summary.main(['--output-root', str(self.root)]), 0)
        output = json.loads((self.root / 'analysis/staged_summary.json').read_text())
        self.assertEqual(output['verdict'], 'incomplete')


if __name__ == '__main__':
    unittest.main()
