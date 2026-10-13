#!/usr/bin/env python3
"""Bounded initialization gradient screen; no optimizer updates or test answers."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from dsa.data.adapters import stable_json_hash
from dsa.training import pilot
from dsa.training.objectives import LossConfig
from scripts.diagnose_dsa_pilot import gradient_probe
from scripts.smoke_pretrained_dsa import atomic_json

DEFAULT_CONDITIONS = {
 'static': {'recurrence_mode':'standard','write_mode':'static'},
 'learned': {'recurrence_mode':'standard','write_mode':'learned'},
 'flt_static': {'recurrence_mode':'flt_memory','write_mode':'static'},
 'flt_learned': {'recurrence_mode':'flt_memory','write_mode':'learned'},
}

def screen(prepared_dir, output_dir, *, device='cuda', smoke_report=None, conditions=None,
           limit=1, depths=(1,8,16), seed=7, workspace=4):
    if limit<1 or limit>4 or not depths or any(k<1 or k>16 for k in depths):
        raise ValueError('Screen is bounded to 1–4 training rows and depths 1–16')
    conditions=conditions or DEFAULT_CONDITIONS
    prepared_dir,output_dir=Path(prepared_dir),Path(output_dir)
    manifest=json.loads((prepared_dir/'manifest.json').read_text())
    config=pilot.PilotConfig(prepared_dir,output_dir/'unused',device=device,tiny=manifest['tiny'],
        backbone=manifest['backbone'],revision=manifest['revision'],loops=manifest['loop_budget'],
        seed=seed,workspace=workspace,smoke_report=smoke_report)
    dev=pilot._device(device);pilot._seed(seed,dev);torch.set_num_threads(1)
    _,manifest_sha,tokenizer,records=pilot._prepared(config)
    selected=[records[i] for i in range(min(limit,len(records)))]
    identity={'prepared_manifest_sha256':manifest_sha,'runtime':pilot._runtime(dev),
              'script_sha256':pilot.sha256_file(__file__),
              'diagnostic_sha256':pilot.sha256_file(ROOT/'scripts/diagnose_dsa_pilot.py'),
              'conditions':conditions,'sample_ids':[x['sample_id'] for x in selected],
              'depths':list(depths),'seed':seed,'workspace':workspace}
    # Commit and dirty state do not change the computation; source hashes do.
    identity['runtime'].pop('git_commit',None);identity['runtime'].pop('working_tree_dirty',None)
    report_path=output_dir/'report.json'
    if report_path.is_file():
        saved=json.loads(report_path.read_text())
        if saved['identity']!=identity:raise ValueError('Screen identity changed; use a new experiment name')
        for name,digest in saved['artifacts_sha256'].items():
            if pilot.sha256_file(output_dir/name)!=digest:raise ValueError('Screen case checksum differs')
        if not saved['finite'] or not saved['model_state_unchanged']:raise RuntimeError('Saved initialization screen failed')
        return saved
    model=pilot._initial_model(config,tokenizer,dev)
    initial=pilot._state_digest(model)
    cases={}
    for label,condition in conditions.items():
        if Path(label).name!=label:raise ValueError('Invalid condition label')
        model.config=replace(model.config,**condition)
        cases[label]=[]
        for index,record in enumerate(selected):
            for budget in depths:
                path=output_dir/label/f'sample{index:04d}-k{budget}.json'
                if path.is_file():
                    result=json.loads(path.read_text())
                    if result['identity_sha256']!=stable_json_hash(identity):raise ValueError('Screen case identity differs')
                else:
                    print(f'Initialization screen: {label}, row={index}, K={budget}',flush=True)
                    result=gradient_probe(model,record,record['latent_answer_prefix_ids'],budget,LossConfig(),seed+index)
                    result['identity_sha256']=stable_json_hash(identity)
                    atomic_json(path,result)
                cases[label].append(result)
    unchanged=pilot._state_digest(model)==initial
    finite=all(g['losses']['answer']['parameter_gradients']['nonfinite_parameter_gradients']==0
               and g['losses']['answer']['parameter_gradients']['global_norm_before_clipping'] is not None
               and math.isfinite(g['losses']['answer']['parameter_gradients']['global_norm_before_clipping'])
               and all(x['finite'] for x in g['raw_residuals']) for values in cases.values() for g in values)
    def median_norm(values, budget):
        norms=[g['losses']['answer']['parameter_gradients']['global_norm_before_clipping'] for g in values if g['loop_budget']==budget]
        return statistics.median(norms) if all(x is not None and math.isfinite(x) for x in norms) else None
    medians={label:{str(k):median_norm(values,k) for k in depths} for label,values in cases.items()}
    ratios={mode:{str(k):medians['flt_'+mode][str(k)]/medians[mode][str(k)] if medians[mode][str(k)] and medians['flt_'+mode][str(k)] is not None else None
                  for k in depths} for mode in ('static','learned') if mode in medians and 'flt_'+mode in medians}
    report={'identity':identity,'complete':True,'finite':finite,'model_state_unchanged':unchanged,
            'optimizer_updates':0,'initial_state_sha256':initial,'parameter_count':sum(p.numel() for p in model.parameters()),
            'median_answer_gradient_norms':medians,'flt_to_standard_ratios':ratios,
            'artifacts_sha256':{str(p.relative_to(output_dir)):pilot.sha256_file(p) for p in output_dir.rglob('sample*.json')},
            'interpretation':'Finite checks are required. A gradient reduction is descriptive; this small screen does not prove stability or select a scientific winner.'}
    atomic_json(report_path,report)
    if not finite or not unchanged:raise RuntimeError('Initialization screen failed finite/model-preservation checks')
    print(json.dumps({'gradient_norms':medians,'flt_to_standard':ratios},indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--smoke-report',type=Path)
    p.add_argument('--conditions',type=json.loads);p.add_argument('--limit',type=int,default=1)
    p.add_argument('--depths',nargs='+',type=int,default=[1,8,16]);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--workspace',type=int,default=4)
    report=screen(**vars(p.parse_args()))
    if not report['finite'] or not report['model_state_unchanged']:raise RuntimeError('Saved screen failed checks')
