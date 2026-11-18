#!/usr/bin/env python3
"""Validate 1-clean+60-attack coverage and build publication-ready CSV/PDF reports.

This script never invents failed/missing observations.  Partial protocols get
an INCOMPLETE coverage audit and a non-zero exit status.  Unsupported native
baseline capacity is disclosed as NOT APPLICABLE, not treated as robustness 0.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from qr64_certified.proposed.iwt_schur_v7 import attacks
from qr64_certified.baselines.registry import ALL_BASELINE_IDS, get_baseline_spec


def load_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def save_csv(path, rows, fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def number(v):
    try:
        n=float(v)
        return n if math.isfinite(n) else None
    except (ValueError,TypeError):return None


def mean(values):
    a=[x for x in values if x is not None]
    return float(np.mean(a)) if a else None


def metric(rows,key): return mean([number(row.get(key)) for row in rows])


def make_plot(paths, output):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed: CSV reports produced, plots skipped.')
        return False
    if not paths:return False
    rows=[]
    for method,group in paths.items():
        attacked=[r for r in group if r['attack']!='none' and r.get('status')=='ok']
        if not attacked:continue
        rows.append((method,mean([number(r.get('attacked_nc')) for r in attacked])))
    rows=[(k,v) for k,v in rows if v is not None]
    if not rows:return False
    rows.sort(key=lambda kv:kv[1],reverse=True)
    fig,ax=plt.subplots(figsize=(max(8,len(rows)*.58),5.2))
    ax.bar(range(len(rows)),[v for _,v in rows],edgecolor='black',linewidth=.45)
    ax.set_xticks(range(len(rows)),[k.replace('dct_','').replace('_',' ') for k,_ in rows],rotation=65,ha='right',fontsize=7)
    ax.set_ylabel('Mean attacked NC (successful decodes)')
    ax.set_ylim(0,1.02)
    ax.set_title('Robustness by method (check coverage before comparison)')
    ax.grid(axis='y',alpha=.25)
    fig.tight_layout()
    fig.savefig(output/'mean_attacked_nc.pdf',bbox_inches='tight')
    fig.savefig(output/'mean_attacked_nc.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return True



def additional_figures(out, benchmark_rows, ablation_rows):
    """Robustness heatmap and ablation effect; all plotted values are observed."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made=[]
    attacks_list=list(dict.fromkeys(c['attack'] for c in attacks.canonical_conditions()[1:]))
    if benchmark_rows:
        matrix=np.full((len(attacks_list),3),np.nan)
        for i,name in enumerate(attacks_list):
            for j,level in enumerate(attacks.LEVELS):
                values=[number(r.get('attacked_nc')) for r in benchmark_rows
                        if r.get('status')=='ok' and r.get('attack')==name
                        and r.get('level')==level]
                matrix[i,j]=mean(values) if any(v is not None for v in values) else np.nan
        fig,ax=plt.subplots(figsize=(8,9))
        rendered=ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=1,aspect='auto',cmap='Greys')
        ax.set_yticks(range(len(attacks_list)),attacks_list,fontsize=8)
        ax.set_xticks(range(3),attacks.LEVELS)
        ax.set_title('Proposed: attacked NC by condition (held-out mean)')
        fig.colorbar(rendered,ax=ax,label='NC',shrink=.70)
        fig.tight_layout()
        for ext in ('pdf','png'):
            path=out/f'proposed_robustness_heatmap.{ext}'
            fig.savefig(path,dpi=180,bbox_inches='tight');made.append(path.name)
        plt.close(fig)
    if ablation_rows:
        names=[r['variant'] for r in ablation_rows]
        values=[r['variant_minus_full_mean_nc'] for r in ablation_rows]
        fig,ax=plt.subplots(figsize=(max(8,len(names)*.65),4.6))
        ax.bar(range(len(names)),values,edgecolor='black',linewidth=.5)
        ax.set_xticks(range(len(names)),[n.replace('_',' ') for n in names],rotation=55,ha='right',fontsize=8)
        ax.axhline(0,linewidth=.9)
        ax.set_ylabel('Mean NC difference from full method')
        ax.set_title('Ablation effect (same host, watermark and attack)')
        ax.grid(axis='y',alpha=.2)
        fig.tight_layout()
        for ext in ('pdf','png'):
            path=out/f'ablation_effect.{ext}'
            fig.savefig(path,dpi=180,bbox_inches='tight');made.append(path.name)
        plt.close(fig)
    return made

def analyze(root, profile, include_baselines, out):
    full=profile=='full'
    expected_count=60 if full else 3
    all_conditions=attacks.canonical_conditions()
    expected=all_conditions[:expected_count+1]
    expected_ids={c['condition_id'] for c in expected}
    dictionary={c['condition_id']:c for c in expected}
    out.mkdir(parents=True,exist_ok=True)
    source_names=['heldout_benchmark','heldout_ablation','development_sweep']
    if full:source_names+=['development_payload']
    if include_baselines:source_names+=['heldout_baselines']
    by_source={}; coverage=[]; all_attacks=[]; failures=[]; group_rows={}
    problems=[]
    for source in source_names:
        path=root/source/'per_attack.csv'
        if not path.exists():
            problems.append(f'Missing experiment CSV: {path}')
            coverage.append({'experiment':source,'status':'missing_file','expected_conditions':expected_count+1})
            continue
        records=load_csv(path);by_source[source]=records
        grouped=defaultdict(list)
        for row in records:
            k=(row.get('method_id',''),row.get('variant',''),row.get('host',''),row.get('watermark',''))
            grouped[k].append(row)
        expected_hosts=16 if source.startswith('heldout_') and full else 1 if source.startswith('heldout_') else 4 if full else 1
        if source=='heldout_benchmark': expected_methods=1
        elif source=='heldout_ablation':expected_methods=13 if full else 2
        elif source=='heldout_baselines':expected_methods=len(ALL_BASELINE_IDS)
        elif source=='development_sweep':expected_methods=9
        elif source=='development_payload':expected_methods=6
        else:expected_methods=0
        if len(grouped)!=expected_hosts*expected_methods:
            problems.append(f'{source}: {len(grouped)} method/pair groups, expected {expected_hosts*expected_methods}')
        for k, rs in sorted(grouped.items()):
            by_case=defaultdict(list)
            for row in rs:
                by_case[row.get('condition_id')].append(row)
            missing=expected_ids-set(by_case)
            unexpected=set(by_case)-expected_ids
            duplicates=sum(len(v)-1 for v in by_case.values())
            mismatched=0
            for condition_id,items in by_case.items():
                if condition_id not in dictionary:continue
                definition=dictionary[condition_id]
                for row in items:
                    if any(row.get(f)!=definition[f] for f in ('attack','level','category','parameter')):
                        mismatched+=1
                    if (row.get('protocol_version')!=attacks.PROTOCOL_VERSION or
                        row.get('protocol_sha256')!=attacks.protocol_sha256()):
                        mismatched+=1
            na=sum(r.get('status')=='not_applicable_capacity' for r in rs)
            errors=[r for r in rs if r.get('status') not in ('ok','not_applicable_capacity')]
            eligible=(source=='heldout_baselines' and get_baseline_spec(k[0]).common_4096_payload) if source=='heldout_baselines' else True
            expected_na=not eligible
            if na and not expected_na:problems.append(f'{source} / {k}: unexpected capacity exclusion')
            if expected_na and na!=expected_count+1:problems.append(f'{source} / {k}: incomplete exclusion records')
            success=not (missing or unexpected or duplicates or mismatched or errors) and (na==expected_count+1 if expected_na else na==0)
            state='ineligible_capacity' if success and expected_na else 'complete' if success else 'incomplete'
            coverage.append({'experiment':source,'method_id':k[0],'variant':k[1],
                'host':k[2],'watermark':k[3],'expected_conditions':expected_count+1,
                'recorded_conditions':len(rs),'ok':sum(r.get('status')=='ok' for r in rs),
                'not_applicable_capacity':na,'errors':len(errors),'missing':len(missing),
                'unexpected':len(unexpected),'duplicates':duplicates,'protocol_mismatch':mismatched,
                'status':state})
            if not success:
                problems.append(f'{source} / {k}: {state}; missing={len(missing)} errors={len(errors)} duplicates={duplicates} mismatch={mismatched}')
            failures.extend({**r,'source':source} for r in errors)
            if eligible:
                group_rows[(source,*k)]=rs
                all_attacks+= [{**r,'source':source} for r in rs if r.get('status')=='ok']
    # Non-attack evidence is checked separately; it is NEVER relabelled as
    # another set of 61 attack conditions.
    aux_checks={}
    auxiliary=[('energy_survey/energy_by_host_band.csv',144),
               ('baseline_audit/paper_baseline_audit.csv',8)]
    if include_baselines:
        auxiliary.append(('paired_statistics/paired_statistics.csv',2*15))
    if full:auxiliary.append(('heldout_security/security_controls.csv',16*(3+24+7)))
    for relative,expected_rows in auxiliary:
        path=root/relative
        observed=load_csv(path) if path.is_file() else []
        aux_checks[relative]={'expected_rows':expected_rows,'observed_rows':len(observed),
                              'exists':path.is_file()}
        if len(observed)!=expected_rows:
            problems.append(f'{relative}: observed {len(observed)} rows, expected {expected_rows}')
        if any(row.get('status')=='error' for row in observed):
            problems.append(f'{relative}: contains errored rows')
    manifest=root/'attack_definitions'/'attack_definitions.csv'
    if not manifest.exists():problems.append(f'Missing manifest {manifest}')
    else:
        definitions=load_csv(manifest)
        if [r.get('condition_id') for r in definitions]!=[r['condition_id'] for r in expected]:
            problems.append(f'Attack manifest does not match {expected_count+1} canonical conditions')
        if any(r.get('protocol_sha256')!=attacks.protocol_sha256() for r in definitions):
            problems.append('Attack manifest protocol hash mismatch')
    overview=[]; category=[]; clean=[]; worst=[]; charts={}
    for (source,method,variant,h,w),rs in sorted(group_rows.items()):
        good=[r for r in rs if r.get('status')=='ok']
        attacking=[r for r in good if r.get('attack')!='none']
        clean_rows=[r for r in good if r.get('attack')=='none']
        overview.append({'experiment':source,'method_id':method,'variant':variant,
            'host':h,'watermark':w,'attacks_ok':len(attacking),
            'attacks_expected':expected_count,
            'mean_nc':metric(attacking,'attacked_nc'),'min_nc':min((number(r.get('attacked_nc')) for r in attacking if number(r.get('attacked_nc')) is not None),default=None),
            'mean_ber':metric(attacking,'attacked_ber'),
            'nc_above_085':sum(number(r.get('attacked_nc')) is not None and number(r.get('attacked_nc'))>.85 for r in attacking),
            'detected_rate':mean([float(r['detected'].strip().lower()=='true') for r in attacking if r.get('detected') in ('True','False')]),
            'clean_psnr_db':number(clean_rows[0].get('clean_psnr_db')) if clean_rows else None,
            'clean_ssim':number(clean_rows[0].get('clean_ssim')) if clean_rows else None,
            'clean_nc':number(clean_rows[0].get('attacked_nc')) if clean_rows else None,
            'clean_ber':number(clean_rows[0].get('attacked_ber')) if clean_rows else None})
        if clean_rows:clean.extend({**r,'source':source} for r in clean_rows)
        for cat in sorted({r['category'] for r in attacking}):
            part=[r for r in attacking if r['category']==cat]
            category.append({'experiment':source,'method_id':method,'variant':variant,
                'host':h,'watermark':w,'category':cat,'n':len(part),
                'mean_nc':metric(part,'attacked_nc'),'mean_ber':metric(part,'attacked_ber')})
        worst.extend([{**r,'source':source} for r in sorted(attacking,key=lambda r:number(r.get('attacked_nc')) if number(r.get('attacked_nc')) is not None else 2)[:3]])
        if source in ('heldout_benchmark','heldout_baselines'):
            charts.setdefault(method,[]).extend(rs)
    # Host-paired, macro-averaged reporting: one host/watermark pair is one unit.
    method_groups=defaultdict(list)
    for r in overview:
        method_groups[(r['experiment'],r['method_id'],r['variant'])].append(r)
    method_summary=[]
    for (source,method,variant),records in sorted(method_groups.items()):
        spec=get_baseline_spec(method) if source=='heldout_baselines' else None
        method_summary.append({'experiment':source,'method_id':method,'variant':variant,
            'blindness_tier':spec.blindness_tier if spec else 'key_assisted_blind',
            'primary_blind_eligible':spec.primary_blind_eligible if spec else True,
            'paper_fidelity_tier':spec.fidelity_tier if spec else 'proposed',
            'host_watermark_pairs':len(records),'attacks_ok':sum(r['attacks_ok'] for r in records),
            'mean_nc_macro':mean([r['mean_nc'] for r in records]),
            'mean_ber_macro':mean([r['mean_ber'] for r in records]),
            'min_nc_all':min((r['min_nc'] for r in records if r['min_nc'] is not None),default=None),
            'mean_clean_psnr_db':mean([r['clean_psnr_db'] for r in records]),
            'mean_clean_ssim':mean([r['clean_ssim'] for r in records]),
            'mean_clean_nc':mean([r['clean_nc'] for r in records]),
            'mean_detection_rate':mean([r['detected_rate'] for r in records])})
    paired_baseline=[]
    proposal={ (r['host'],r['watermark'],r['condition_id']):r
               for r in by_source.get('heldout_benchmark',[]) if r.get('status')=='ok' }
    if include_baselines:
        for method in ALL_BASELINE_IDS:
            spec=get_baseline_spec(method)
            eligible=[r for r in by_source.get('heldout_baselines',[])
                      if r.get('method_id')==method and r.get('status')=='ok' and r.get('attack')!='none']
            matched=[(proposal.get((r['host'],r['watermark'],r['condition_id'])),r) for r in eligible]
            matched=[(p,b) for p,b in matched if p and p.get('status')=='ok']
            delta_nc=[number(p.get('attacked_nc'))-number(b.get('attacked_nc'))
                      for p,b in matched if number(p.get('attacked_nc')) is not None
                      and number(b.get('attacked_nc')) is not None]
            delta_ber=[number(p.get('attacked_ber'))-number(b.get('attacked_ber'))
                       for p,b in matched if number(p.get('attacked_ber')) is not None
                       and number(b.get('attacked_ber')) is not None]
            paired_baseline.append({'baseline_id':method,'blindness_tier':spec.blindness_tier,
                'fidelity_tier':spec.fidelity_tier,'capacity_4096_supported':spec.common_4096_payload,
                'primary_blind_eligible':spec.primary_blind_eligible,
                'matched_attack_rows':len(matched),'expected_attack_rows':expected_count*(16 if full else 1),
                'proposal_minus_baseline_mean_nc':mean(delta_nc),
                'proposal_minus_baseline_mean_ber':mean(delta_ber),
                'status':'ineligible_capacity' if not spec.common_4096_payload else
                         'complete' if len(matched)==expected_count*(16 if full else 1) else 'incomplete'})
    ablation=[]
    full_by_pair={ (r.get('host'),r.get('watermark'),r.get('condition_id')):r
        for r in by_source.get('heldout_ablation',[])
        if r.get('variant')=='full' and r.get('status')=='ok' }
    by_variant=defaultdict(list)
    for r in by_source.get('heldout_ablation',[]):
        if r.get('status')!='ok' or r.get('attack')=='none' or r.get('variant')=='full':continue
        ref=full_by_pair.get((r.get('host'),r.get('watermark'),r.get('condition_id')))
        if not ref:continue
        p=number(ref.get('attacked_nc'));v=number(r.get('attacked_nc'))
        if p is not None and v is not None:by_variant[r['variant']].append(v-p)
    for variant,values in sorted(by_variant.items()):
        ablation.append({'variant':variant,
            'ablation_type':'decoder_only' if variant in ('direct','one_pilot','no_geometry','no_photo','no_denoise','no_structure') else 'reembedding',
            'matched_attack_rows':len(values),'variant_minus_full_mean_nc':mean(values),
            'variant_minus_full_median_nc':float(np.median(values))})
    security_summary=[]
    for r in by_source.get('heldout_security',[]):
        security_summary.append({'host':r.get('host'),'watermark':r.get('watermark'),
            'control':r.get('control'),'status':r.get('status'),
            'detected':r.get('detected'),'pilot_accuracy':r.get('pilot_accuracy'),
            'phase_accuracy':r.get('phase_accuracy')})
    negative=[r for r in security_summary if r['control']!='positive' and r['status']=='ok']
    security_overview=[{'n_positive':sum(r['control']=='positive' and r['status']=='ok' for r in security_summary),
        'positive_detected':sum(r['control']=='positive' and r['detected']=='True' for r in security_summary),
        'n_negative':len(negative),'negative_false_accepts':sum(r['detected']=='True' for r in negative),
        'false_positive_rate':mean([float(r['detected']=='True') for r in negative])}]
    save_csv(out/'method_summary.csv',method_summary)
    save_csv(out/'paired_baseline_comparison.csv',paired_baseline)
    save_csv(out/'ablation_delta.csv',ablation)
    save_csv(out/'security_controls.csv',security_summary)
    save_csv(out/'security_summary.csv',security_overview)
    save_csv(out/'coverage.csv',coverage)
    save_csv(out/'clean.csv',clean)
    save_csv(out/'per_host_summary.csv',overview)
    save_csv(out/'per_category.csv',category)
    save_csv(out/'worst_attacks.csv',worst)
    save_csv(out/'failures.csv',failures)
    plotted=make_plot(charts,out)
    extra_figures=additional_figures(out,by_source.get('heldout_benchmark',[]),ablation)
    report={'status':'complete' if not problems else 'incomplete',
        'protocol_version':attacks.PROTOCOL_VERSION,
        'protocol_sha256':attacks.protocol_sha256(),
        'profile':profile,'expected_per_method_host_watermark':expected_count+1,
        'expected_attacks':expected_count,'clean_count':1,
        'source_names':source_names,'coverage_groups':len(coverage),
        'complete_groups':sum(c.get('status')=='complete' for c in coverage),
        'ineligible_capacity_groups':sum(c.get('status')=='ineligible_capacity' for c in coverage),
        'charts_generated':plotted,'additional_figures':extra_figures,
        'auxiliary_checks':aux_checks,'problems':problems}
    (out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('PROTOCOL AUDIT:',report['status'],f'{report["complete_groups"]} complete groups',
          f'{report["ineligible_capacity_groups"]} capacity exclusions',f'{len(problems)} problems',flush=True)
    for problem in problems[:12]:print('  -',problem,flush=True)
    return not problems


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'outputs/paper_protocol')
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--profile',choices=('quick','full'),default='full')
    p.add_argument('--include-baselines',action='store_true')
    a=p.parse_args()
    root=a.root.resolve();out=(a.out or root/'reports').resolve()
    if not analyze(root,a.profile,a.include_baselines,out):raise SystemExit(2)
if __name__=='__main__':main()
