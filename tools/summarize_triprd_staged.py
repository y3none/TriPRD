#!/usr/bin/env python3
"""Audit paired v5 development reports; never start final or claim significance."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

METHOD = 'triprd_staged_v5'
GROUPS = ('S0', 'S1', 'S2', 'S3')
METRICS = ('S_alpha', 'F_beta_w', 'E_phi', 'MAE')
MATCHED = ('model_variant', 'base_channels', 'image_size', 'pretrained', 'train_backbone_blocks',
    'detail_branch', 'train_seed', 'split_seed', 'batch_size', 'num_workers', 'lr',
    'backbone_lr_scale', 'weight_decay', 'warmup_epochs', 'amp', 'grad_clip', 'augmentation',
    'boundary_weight', 'aux_weight', 'teacher_epochs', 'source_epochs', 'epochs', 'val_metrics_every',
    'generic_weights_sha256', 'generic_config_sha256', 'dino_source_sha256', 'source_sha256',
    'prediction_protocol', 'cod_optimizer_reset', 'cod_training_rng_reset')
RULE = ('Predeclared conservative single-seed screen: at each group\'s S_alpha-selected epoch, '
    'S3 must strictly improve S_alpha over S0/S1/S2, without reducing F_beta_w or mean E_phi '
    'or increasing MAE; S3 must also win S_alpha against each control at more than half '
    'of all scheduled common validation epochs. This is not statistical significance or SOTA.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label):
    require(type(value) in (float, int) and math.isfinite(value), label + ': missing/nonfinite number')
    return float(value)


def hash_value(value, label):
    require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), label + ': missing SHA256')
    return value


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(value, dict), 'Expected report object')
    return value


def members(report, prefix, empty=False):
    p, counts = report['protocol'], report.get('sample_counts', {})
    result = {}
    for split in ('train', 'val'):
        key = prefix + '_' + split
        values = p.get(key)
        require(isinstance(values, list) and (empty or bool(values)), 'Missing ' + key)
        require(all(isinstance(x, str) for x in values) and len(values) == len(set(values)), 'Duplicate/invalid ' + key)
        require(counts.get(key) == len(values), 'Count differs: ' + key)
        result[key] = values
    require(not set(result[prefix + '_train']) & set(result[prefix + '_val']), prefix + ' train/val overlap')
    return result


def history(report, epochs):
    rows = report.get('history', [])
    require([r.get('epoch') for r in rows] == list(range(1, epochs + 1)), 'Incomplete/duplicate epoch history')
    require(report.get('completed_epochs') == epochs, 'Incomplete budget')
    for row in rows:
        t = row.get('train', {})
        for name in ('steps', 'optimizer_steps', 'amp_skipped_steps', 'primary_images', 'source_images'):
            require(type(t.get(name)) is int and t[name] >= 0, 'Missing training accounting: ' + name)
        require(t['steps'] > 0 and t['steps'] == t['optimizer_steps'] + t['amp_skipped_steps'], 'Optimizer accounting differs')
        number(t.get('seconds'), 'train.seconds')
        for name, value in t.items():
            if type(value) is float:
                number(value, name)
    return rows


def cost(report):
    rows = report['history']
    fields = ('steps', 'optimizer_steps', 'amp_skipped_steps', 'primary_images', 'source_images', 'seconds')
    return {**{name: sum(row['train'][name] for row in rows) for name in fields},
        'epochs': report['completed_epochs'], 'parameters': report.get('parameters'),
        'peak_cuda_allocated_bytes': max((r['train'].get('peak_cuda_allocated_bytes') or 0 for r in rows), default=0)}


def base(report, group, stage):
    require(report.get('method') == METHOD and report.get('group') == group and report.get('stage') == stage, 'Wrong method/group/stage')
    require(report.get('status') == 'complete', 'Run is incomplete')
    require(report.get('formal') is False and report.get('test_status') == 'skipped' and report.get('test') == {}, 'Development run accessed official tests')
    p = report.get('protocol')
    require(isinstance(p, dict) and digest(p) == report.get('contract_sha256'), 'Contract digest differs')
    hash_value(report.get('checkpoint_sha256'), 'checkpoint_sha256')
    for key in ('generic_initial_state_sha256', 'initial_state_sha256', 'cod_decoder_before_sha256', 'cod_decoder_after_sha256'):
        hash_value(report.get(key), key)
    require(report['cod_decoder_before_sha256'] == report['cod_decoder_after_sha256'], 'COD decoder changed during source transfer')
    require(report.get('prediction_protocol') == p.get('prediction_protocol') and bool(p.get('prediction_protocol')), 'Prediction protocol missing/different')
    require(report.get('stage_exposure') == p.get('stage_exposure') and isinstance(p.get('stage_exposure'), dict), 'Missing/different stage exposure')
    return p


def validation(value, val_members):
    count = len(val_members)
    require(isinstance(value, dict) and value.get('num_samples') == count, 'Missing/incorrect validation count')
    sources = value.get('sources', {})
    require(set(sources) == {'COD10K', 'CAMO'}, 'Validation must contain both COD sources')
    expected = Counter(key.split('/', 1)[0] for key in val_members)
    require({name: x.get('num_samples') for name, x in sources.items()} == dict(expected), 'Source validation counts differ from member identities')
    result = {k: number(value.get('macro', {}).get(k), 'macro.' + k) for k in METRICS}
    for key in METRICS:
        mean = sum(number(s.get(key), key) for s in sources.values()) / 2
        require(math.isclose(mean, result[key], rel_tol=0, abs_tol=1e-12), 'Macro/source metrics differ: ' + key)
    return result


def screen(report, group):
    p = base(report, group, 'cod')
    c = report.get('config', {})
    require(report.get('selection_metric') == 'macro_S_alpha' and c.get('cod_train_mode') == 'split', 'Not an internal screen')
    epochs, every = c.get('epochs'), c.get('val_metrics_every')
    require(type(epochs) is int and epochs > 0 and type(every) is int and every > 0, 'Missing validation/COD budget')
    m = members(report, 'cod')
    unused = members(report, 'vdt', empty=True)
    require(not any(unused.values()), 'COD stage accessed VDT')
    rows = history(report, epochs)
    require(p.get('cod_optimizer_reset') is True and p.get('cod_training_rng_reset') is True, 'COD optimizer/training RNG must reset')
    require(p['stage_exposure'] == {'optimization': ['COD_train'], 'selection': ['COD_val'], 'teacher_in_stage': False, 'source_batches_in_cod': 0}, 'COD exposure differs')
    require(all(r['train']['source_images'] == 0 and r['train']['primary_images'] == len(m['cod_train']) for r in rows), 'COD training exposure differs')
    vals = {r['epoch']: validation(r['validation'], m['cod_val']) for r in rows if 'validation' in r}
    expected = [e for e in range(1, epochs + 1) if e % every == 0 or e == epochs]
    require(list(vals) == expected, 'Validation epoch grid incomplete/changed')
    chosen = max(vals, key=lambda e: vals[e]['S_alpha'])
    require(chosen == report.get('best_epoch'), 'Selected epoch is not first maximum S_alpha')
    require(math.isclose(number(report.get('best_score'), 'best_score'), vals[chosen]['S_alpha'], rel_tol=0, abs_tol=1e-12), 'Selected score differs from history')
    for key in MATCHED:
        require(key in p, 'Missing matched protocol field: ' + key)
        if key not in ('generic_weights_sha256', 'generic_config_sha256', 'dino_source_sha256'):
            require(p[key] is not None, 'Null matched protocol field: ' + key)
        if key in c:
            require(c[key] == p[key], 'Config/protocol differ: ' + key)
    require(isinstance(p['source_sha256'], dict) and bool(p['source_sha256']), 'Missing source file hashes')
    for name, value in p['source_sha256'].items():
        hash_value(value, 'source file ' + name)
    if p['model_variant'] != 'tiny':
        require(p['pretrained'] is True, 'Real model must use declared generic pretraining')
        hash_value(p['generic_weights_sha256'], 'generic weights')
        hash_value(p['dino_source_sha256'], 'DINO source')
    return {'group': group, 'selected_epoch': chosen, 'selected_metrics': vals[chosen],
        'selected_sources': rows[chosen - 1]['validation']['sources'], 'validation': vals,
        'cost': cost(report), 'checkpoint_sha256': report['checkpoint_sha256']}


def source(report, group, cod):
    p = base(report, group, 'source')
    epochs = cod['protocol']['source_epochs']
    require(report.get('selection_metric') == 'source_validation_loss', 'Wrong source selection')
    require(report.get('source_provenance') is None and p.get('source_provenance') is None, 'Source stage must start without a prior source checkpoint')
    m = members(report, 'vdt')
    require(not any(members(report, 'cod', empty=True).values()), 'Source stage accessed COD')
    require(p['stage_exposure'] == {'optimization': ['VDT_train'], 'selection': ['VDT_val'], 'teacher_in_stage': group in ('S2', 'S3'), 'source_batches_in_cod': 0}, 'Source exposure differs')
    rows = history(report, epochs)
    require(all(r.get('validation', {}).get('objective') == 'supervised_gt_only' and r['validation'].get('num_samples') == len(m['vdt_val']) for r in rows), 'Source selection must use supervised GT validation')
    require(all(r['train']['primary_images'] == len(m['vdt_train']) and r['train']['source_images'] == len(m['vdt_train']) for r in rows), 'Source exposure accounting differs')
    chosen = min(rows, key=lambda r: number(r['validation'].get('loss'), 'source.loss'))
    require(chosen['epoch'] == report.get('best_epoch') and math.isclose(-chosen['validation']['loss'], number(report.get('best_score'), 'source.best_score'), rel_tol=0, abs_tol=1e-12), 'Source selection differs from history')
    prov = cod.get('source_provenance')
    require(isinstance(prov, dict) and prov == cod['protocol'].get('source_provenance'), 'Missing/different source provenance')
    for key in ('checkpoint_sha256', 'contract_sha256', 'group', 'completed_epochs', 'generic_initial_state_sha256', 'initial_state_sha256'):
        require(prov.get(key) == report.get(key), 'Source provenance mismatch: ' + key)
    require(prov.get('contract') == p and prov.get('epoch') == chosen['epoch'] and prov.get('transfer_scope') == 'shared_only_exclude_decoders', 'Source provenance contract/selection/scope differs')
    hash_value(prov.get('completed_checkpoint_sha256'), 'completed source checkpoint')
    require(report['initial_state_sha256'] == report['generic_initial_state_sha256'], 'Source did not start from generic initialization')
    for key in ('generic_initial_state_sha256', 'cod_decoder_before_sha256', 'cod_decoder_after_sha256'):
        require(report[key] == cod[key], 'Source/COD initialization mismatch: ' + key)
    for key in MATCHED:
        require(p.get(key) == cod['protocol'][key], 'Source/COD protocol differs: ' + key)
    return {'group': group, 'selected_epoch': chosen['epoch'], 'selected_gt_loss': chosen['validation']['loss'], 'cost': cost(report), 'provenance': prov}


def teacher_cost(root, source_report, mode):
    config = source_report.get('config', {})
    candidates = [root / 'teachers' / mode / 'comparison.json']
    if config.get('teacher_checkpoint'):
        candidates.append(Path(config['teacher_checkpoint']).parent / 'comparison.json')
    path = next((x for x in candidates if x.is_file()), None)
    require(path is not None, 'Missing ' + mode + ' teacher report for cost/provenance audit')
    r = read(path)
    require(r.get('method') == 'triprd_task_v4' and r.get('stage') == 'teacher' and r.get('teacher_mode') == mode and r.get('status') == 'complete', 'Invalid teacher report')
    require(digest(r.get('protocol')) == r.get('contract_sha256'), 'Teacher contract digest differs')
    require(r.get('checkpoint_sha256') == source_report['protocol'].get('teacher_sha256'), 'Teacher dependency hash differs')
    require(r.get('test_status') == 'skipped' and r.get('test') == {}, 'Teacher accessed official tests')
    require(all(r.get('sample_counts', {}).get(key) == 0 and r['protocol'].get(key, []) == [] for key in ('cod_train', 'cod_val')), 'Teacher accessed COD or lacks zero-count evidence')
    vm = members(r, 'vdt')
    require(all(vm[k] == source_report['protocol'][k] for k in vm), 'Teacher/source VDT members differ')
    history(r, r['config']['teacher_epochs'])
    return {'report_path': str(path.resolve()), 'checkpoint_sha256': r['checkpoint_sha256'], 'cost': cost(r)}


def summarize(output_root):
    root = Path(output_root)
    result = {'method': METHOD, 'output_root': str(root.resolve()), 'verdict': 'incomplete',
        'report_consistency_verified': False, 'rule': RULE, 'errors': [], 'screen': {}, 'source': {},
        'teachers': {}, 'same_epoch': [], 'selected_deltas': {}, 'same_epoch_wins': {},
        'checkpoint_files_checked': [], 'checkpoint_files_missing': [],
        'limitations': ['One paired training seed; no statistical significance.',
            'Internal VAL only; no official COD/SOTA comparison.',
            'E_phi denotes mean E, not adaptive E.',
            'Seconds are measured runtime, not hardware-normalized FLOPs; teacher cost is additional.',
            'Report consistency does not certify unavailable checkpoint or dataset bytes.']}
    reports, source_reports = {}, {}
    for group in GROUPS:
        path = root / 'screen' / group / 'comparison.json'
        try:
            r = reports[group] = read(path)
            result['screen'][group] = screen(r, group)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result['errors'].append(f'screen/{group}: {exc}')
    for group in GROUPS[1:]:
        try:
            r = source_reports[group] = read(root / 'source' / group / 'comparison.json')
            result['source'][group] = source(r, group, reports[group])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result['errors'].append(f'source/{group}: {exc}')
    try:
        reference = reports['S0']
        require(reference.get('source_provenance') is None and reference['protocol'].get('source_provenance') is None, 'S0 must have no source provenance')
        require(reference['initial_state_sha256'] == reference['generic_initial_state_sha256'], 'S0 did not start from generic initialization')
        for group, r in reports.items():
            for key in (*MATCHED, 'cod_train', 'cod_val'):
                require(r['protocol'].get(key) == reference['protocol'].get(key), group + ' unmatched protocol: ' + key)
            for key in ('generic_initial_state_sha256', 'cod_decoder_before_sha256', 'cod_decoder_after_sha256'):
                require(r.get(key) == reference.get(key), group + ' unmatched initialization: ' + key)
        for group, r in source_reports.items():
            for key in ('vdt_train', 'vdt_val'):
                require(r['protocol'][key] == source_reports['S1']['protocol'][key], 'Source groups use different ' + key)
        for group, mode in (('S2', 'rgb'), ('S3', 'multimodal')):
            result['teachers'][mode] = teacher_cost(root, source_reports[group], mode)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['errors'].append('paired protocol: ' + str(exc))
    for stage, collection in (('screen', reports), ('source', source_reports)):
        for group, r in collection.items():
            path = root / stage / group / 'best.pt'
            if not path.is_file():
                result['checkpoint_files_missing'].append(str(path))
            elif file_digest(path) != r.get('checkpoint_sha256'):
                result['errors'].append(f'{stage}/{group}: actual checkpoint hash differs')
            else:
                result['checkpoint_files_checked'].append(str(path))
    if result['errors']:
        return result
    rows = result['screen']
    epochs = list(rows['S0']['validation'])
    for epoch in epochs:
        result['same_epoch'].append({'epoch': epoch, 'metrics': {g: rows[g]['validation'][epoch] for g in GROUPS},
            'S3_minus_controls': {g: {k: rows['S3']['validation'][epoch][k] - rows[g]['validation'][epoch][k] for k in METRICS} for g in GROUPS[:3]}})
    candidate = rows['S3']['selected_metrics']
    criteria = {}
    for group in GROUPS[:3]:
        control = rows[group]['selected_metrics']
        delta = {key: candidate[key] - control[key] for key in METRICS}
        wins = sum(rows['S3']['validation'][epoch]['S_alpha'] > rows[group]['validation'][epoch]['S_alpha'] for epoch in epochs)
        result['selected_deltas']['S3-' + group] = delta
        result['same_epoch_wins']['S3-' + group] = {'wins': wins, 'evaluations': len(epochs), 'fraction': wins / len(epochs)}
        criteria[group] = {'selected_metrics_pass': delta['S_alpha'] > 0 and delta['F_beta_w'] >= 0 and delta['E_phi'] >= 0 and delta['MAE'] <= 0,
                           'same_epoch_majority_pass': wins > len(epochs) / 2}
    result.update(report_consistency_verified=True, criteria=criteria,
        evidence_scope='smoke_only' if reports['S0']['protocol']['model_variant'] == 'tiny' else 'single_seed_development',
        verdict='provisional_suggests_benefit' if reports['S0']['protocol']['model_variant'] != 'tiny' and all(all(v.values()) for v in criteria.values()) else 'not_supported',
        validation_epochs=epochs, train_seed=reports['S0']['protocol']['train_seed'], split_seed=reports['S0']['protocol']['split_seed'])
    return result


def markdown(result):
    lines = ['# Staged v5 development audit', '', '**Verdict: ' + result['verdict'] + '**', '', result['rule'], '',
        'Evidence scope: ' + result.get('evidence_scope', 'incomplete') + '. Tiny/CPU smoke runs cannot establish model benefit.', '',
        'This report never starts final training. Report consistency verified: ' + str(result['report_consistency_verified']) + '.', '']
    if result['errors']:
        lines += ['## Missing or inconsistent evidence', ''] + ['- ' + e for e in result['errors']] + ['']
    lines += ['## S_alpha-selected checkpoints', '', 'All metrics in each row come from that one selected epoch. E is mean E.', '',
        '| Group | Epoch | S_alpha | F_beta_w | E_mean | MAE |', '|---|---:|---:|---:|---:|---:|']
    for group, row in result['screen'].items():
        lines.append('| ' + group + ' | ' + str(row['selected_epoch']) + ' | ' + ' | '.join(f'{row["selected_metrics"][k]:.6f}' for k in METRICS) + ' |')
    lines += ['', '## S3 minus controls at selected checkpoints', '', '| Comparison | Delta S | Delta Fw | Delta E_mean | Delta MAE | Same-epoch S wins |', '|---|---:|---:|---:|---:|---:|']
    for key, row in result['selected_deltas'].items():
        w = result['same_epoch_wins'][key]
        lines.append('| ' + key + ' | ' + ' | '.join(f'{row[k]:+.6f}' for k in METRICS) + f' | {w["wins"]}/{w["evaluations"]} |')
    lines += ['', '## Every scheduled validation epoch', '', '| Epoch | Group | S_alpha | F_beta_w | E_mean | MAE |', '|---:|---|---:|---:|---:|---:|']
    for row in result['same_epoch']:
        for group in GROUPS:
            lines.append(f'| {row["epoch"]} | {group} | ' + ' | '.join(f'{row["metrics"][group][k]:.6f}' for k in METRICS) + ' |')
    lines += ['', '## Training cost', '', 'Full stage budgets, including epochs after the selected checkpoint; teachers are additional and reusable. Runtime is hardware dependent.', '', '| Stage/group | Epochs | Steps | Updates | Skips | Primary images | Seconds |', '|---|---:|---:|---:|---:|---:|---:|']
    for stage in ('screen', 'source', 'teachers'):
        for group, item in result[stage].items():
            c = item['cost']
            lines.append(f'| {stage}/{group} | {c["epochs"]} | {c["steps"]} | {c["optimizer_steps"]} | {c["amp_skipped_steps"]} | {c["primary_images"]} | {c["seconds"]:.1f} |')
    lines += ['', '## Source provenance', '']
    for group, item in result['source'].items():
        p = item['provenance']
        lines.append(f'- {group}: source epoch {p["epoch"]}/{p["completed_epochs"]}; checkpoint `{p["checkpoint_sha256"]}`; scope `{p["transfer_scope"]}`.')
    lines += ['', '## Limits', ''] + ['- ' + text for text in result['limitations']]
    lines.append(f'- Actual checkpoint files checked: {len(result["checkpoint_files_checked"])}; unavailable: {len(result["checkpoint_files_missing"])}.')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='Output directory (default: OUTPUT_ROOT/analysis)')
    args = parser.parse_args(argv)
    result = summarize(args.output_root)
    out = args.out or args.output_root / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'staged_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (out / 'staged_summary.md').write_text(markdown(result), encoding='utf-8')
    print(json.dumps({'verdict': result['verdict'], 'report_consistency_verified': result['report_consistency_verified'], 'output': str(out.resolve())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
