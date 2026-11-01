# Recompute the public manifest/compiler storage audit from retained metadata.
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
raw=Path(__file__).resolve().parent
reports={}
for contract in sorted(raw.glob('*-layout-contract.json')):
 d=json.loads(contract.read_text()); predicted=d['storage_report']; manifest=d['manifest']['arguments']
 actual=defaultdict(Counter); payload=Counter(); kind_map={'seed_weight':'weight','poly_coeff':'coefficient','mult_scalar_weight':'scalar'}
 level_lengths=defaultdict(set)
 calibration=predicted['calibration']['bytes_per_element']

 def decision(count, capacity, target, enabled):
  legacy=count*calibration[str(target)]
  candidate=(math.ceil(count/capacity)*calibration[str(target+1)]
             if str(target+1) in calibration else None)
  assert enabled == (candidate is not None and candidate < legacy)
  return {'enabled':enabled,'use_site_level':target,'seed_level':target+1,
          'legacy_count':count,'seed_count':math.ceil(count/capacity),
          'legacy_bytes':legacy,'candidate_bytes':candidate,
          'reason':('saves_bytes' if enabled else 'seed_level_unavailable' if candidate is None
                    else 'single_request' if count==1 else 'no_byte_benefit')}
 for a in manifest.values():
  kind=kind_map.get(a['param_kind'],a['param_kind']); level=str(a['level'])
  count=1
  for size in a['size']:count*=size
  actual[a['layer_id']][(level,kind)]+=count
  lengths=[e['length'] for e in a['elements']]
  assert len(lengths)==count
  payload[a['layer_id']]+=sum(lengths)
  level_lengths[level].update(lengths)
 layers=[]
 for layer in predicted['layers']:
  name=layer['layer_id']
  counts=Counter({(lv,k):v for lv,ks in layer['counts'].items() for k,v in ks.items() if v})
  assert counts==actual.pop(name,Counter()),(name,counts)
  assert layer['bytes']==payload[name],(name,layer['bytes'],payload[name])
  record=dict(layer,actual_payload_bytes=payload[name],counts_and_level_match=True)
  graph_layer=d['graph']['layer'][name]
  if layer['type']=='conv2d' and 'convolution_seed_layout' not in layer:
   record['seed_selection_note']='legacy reference' if 'legacy' in contract.name else 'stride does not divide seed shape, or unsupported geometry'
   record['input_geometry']=d['graph']['feature'][graph_layer['feature_input'][0]]
  if layer['type']=='polyact' and 'poly_seed_layout' not in layer:
   record['seed_selection_note']='legacy reference' if 'legacy' in contract.name else 'single channel group, incompatible geometry, or no positive byte benefit'
  if layer['type']=='polyact' and 'poly_seed_layout' not in layer and 'legacy' not in contract.name:
   groups=sum(kinds.get('coefficient',0) for kinds in layer['counts'].values()) // (graph_layer['order']+1)
   if groups == 1:
    record['seed_selection_note']='single channel group: one seed would cost more at the higher level'
  if layer['type']=='mult_scalar':
   record['seed_selection_note']='mult-scalar remains encrypted in its original layout by design'
  if 'dense_seed_tiles' in layer:
   layout=layer['dense_seed_tiles']
   record['kind_decisions']={kind:decision(math.prod(layout['weight_shape']) if kind=='weight' else layout['weight_shape'][0],
                                         layout['group_size'],layout['use_site_level']-(kind=='bias'),layout[kind])
                            for kind in ('weight','bias')}
   record['unpacked_kinds']={kind: ('single request: no byte benefit' if (math.prod(layout['weight_shape']) if kind=='weight' else layout['weight_shape'][0])==1 else 'no positive byte benefit or seed level unavailable')
                            for kind in ('weight','bias') if not layout[kind]}
  if 'poly_seed_layout' in layer:
   layout=layer['poly_seed_layout']
   record['degree_decisions']=[dict(degree=degree,**decision(layout['channel_groups'],layout['group_size'],target,enabled))
                               for degree,(target,enabled) in enumerate(zip(layout['use_site_levels'],layout['packed_degrees']))]
  layers.append(record)
 assert not actual,actual
 assert predicted['storage_bytes']==sum(payload.values())
 assert not predicted['unaccounted'],predicted['unaccounted']
 assert all(lengths=={calibration[lv]} for lv,lengths in level_lengths.items()),level_lengths
 reports[contract.name.removesuffix('-layout-contract.json')]={
  'predicted_payload_bytes':predicted['storage_bytes'],'actual_payload_bytes':sum(payload.values()),
  'all_counts_levels_and_bytes_match':True,'bytes_by_seed_level':{lv:next(iter(v)) for lv,v in level_lengths.items()},
  'eval_context_bytes':d['eval_context_bytes'],'layers':layers}
(raw.parent / 'layout_storage_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
for name,r in reports.items():print(name,r['actual_payload_bytes'],len(r['layers']),'layers: counts/levels/bytes match')
