"""Primary five-seed test comparison with the author's supplied results."""
import json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent
m=json.loads((r/'manifest.json').read_text());groups={'dft':[],'cubelearn':[]}
for t in m['tasks']:
    p=r/'runs'/t['name']
    if not (p/'status.json').exists():
        print('PENDING/INCOMPLETE',t['name']);continue
    assert json.loads((p/'status.json').read_text())['exit_code']==0,t['name']
    v=json.loads((p/'results.json').read_text())
    for k in ['epochs','batch_size','seed','lr','lpp_lr']:assert v[k]==t[k],(t['name'],k)
    assert len(v['history'])==60
    best=max(v['history'],key=lambda x:(x['val_acc'],-x['val_loss']))
    assert v['best_epoch']==best['epoch']+1
    groups[t['arm']].append(v)
    print(t['name'],'best epoch',v['best_epoch'],'seen',round(100*v['test_in']['acc'],2),'heldout',round(100*v['test_out']['acc'],2))
for arm,rows in groups.items():
    if len(rows)!=5:
        print(arm,'incomplete:',len(rows),'/5');continue
    print('\n',arm,'all five seeds, mean ± sample SD')
    for split,label in [('test_in','seen'),('test_out','heldout')]:
        values=[100*v[split]['acc'] for v in rows]
        ref=m['author_results'][arm]
        print(label,f'{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}',f'author {ref[label+"_mean"]:.2f} ± {ref[label+"_sd"]:.2f}')
