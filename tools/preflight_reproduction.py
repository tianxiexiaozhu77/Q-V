#!/usr/bin/env python3
"""Presence-only checks of major external prerequisites; does not load a model."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();root=a.workspace.resolve()
required=[
'baselines/evaluation/matched_phase6_20261022/catalog.json',
'baselines/evaluation/query_transfer_20261022_v1/catalog.json',
'baselines/evaluation/chexchonet_20261023/catalog.json',
'AlphaRAD/local_setup/chexchonet_20261023/catalog.json',
'AlphaRAD/local_setup/chexpert_query_data_20261022_shot5_seed42/catalog.json',
'experiments/query_search_20261021_152010/assets/vision_tokens_fp32',
'experiments/fewshot_multiseed_20261024/protocol_v1/catalog.json',
'experiments/fewshot_multiseed_20261024/cache_reuse_sources_v1.json',
'experiments/rsna_chest_stability_20261024/runs/full_20261024_v1',
'experiments/fewshot_multiseed_20261024/runs/full_20261024_v2',
'experiments/query_path_intervention_20261024/protocol_siim_v1/catalog.json',
'experiments/query_path_multibackbone_20261025/protocol_v1/catalog.json',
'experiments/qata_validation_threshold_20261024/protocol_v1/catalog.json',
'experiments/query_resource_completion_20261103/query_descriptions.json',
'baselines/paper_draft_zh/CHEXCHONET_VERIFIED.json']
result=[{'path':s,'exists':(root/s).exists()} for s in required]
print(json.dumps({'major_prerequisites':result,'scope':'Presence only; hashes, transitive dependencies, data identity and code patches are checked separately by experiment workers.'},indent=2))
raise SystemExit(any(not x['exists'] for x in result))
