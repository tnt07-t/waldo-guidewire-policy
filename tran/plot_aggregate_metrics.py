"""Plot only documented aggregates; never interpolate missing training histories."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / 'selected_metrics.csv').open()))
ids = ['gru-adam', 'gru-adamw', 'gru-early-goal', 'static-latest', 'static-grid', 'lastblock-static']
labels = ['Late-goal GRU\nAdam', 'Late-goal GRU\nAdamW', 'Early-goal GRU\nAdam', 'Summary static\nAdam', '4×4 grid static\nAdam', 'Adapt last block\n20-minute cap']
val = {r['run']: r for r in rows if r['partition'] == 'validation'}
train = {r['run']: r for r in rows if r['partition'] == 'train'}
plt.rcParams.update({'font.size': 12, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})

def save(fig, name, caption):
    fig.text(.03, .025, caption, fontsize=10, va='bottom')
    fig.tight_layout(rect=(0, .12, 1, .95))
    for ext in ('svg', 'png'):
        fig.savefig(ROOT / 'figures' / f'{name}.{ext}', dpi=200, facecolor='white')
    plt.close(fig)

x = np.arange(6)
fig, ax = plt.subplots(figsize=(14, 6))
y = [100*float(val[k]['balanced_accuracy']) for k in ids]
ax.bar(x,y,color=['#245A81']*3+['#258477','#768C39','#9C6AA5'])
for i,v in enumerate(y): ax.text(i,v+1,f'{v:.2f}%',ha='center')
ax.axhline(50,color='#666',linestyle='--',label='Single-class prediction: 50% BA')
ax.set(xticks=x,xticklabels=labels,ylim=(0,100),ylabel='Validation balanced accuracy (%)',title='Frozen-feature runs and time-limited last-block adaptation')
ax.legend(loc='upper right')
save(fig,'validation_balanced_accuracy','Frozen runs: split 0 / seed 0. All reports: 80 validation cases. Single runs; no uncertainty estimate.\nAdaptation: best of 8 saved epochs; overall timeout, reload/frozen checks not saved. Different heads/recipes.')

fig,ax=plt.subplots(figsize=(14,6))
for off,key,color,label in [(-.19,'retract_recall','#B95645','RETRACT (34 cases)'),(.19,'advance_recall','#245A81','ADVANCE (46 cases)')]:
 y=[100*float(val[k][key]) for k in ids];ax.bar(x+off,y,.36,color=color,label=label)
 for i,v in enumerate(y): ax.text(i+off,v+1,f'{v:.1f}',ha='center',fontsize=10)
ax.set(xticks=x,xticklabels=labels,ylim=(0,105),ylabel='Validation recall (%)',title='Similar balanced accuracy can hide different action preferences')
ax.legend(loc='upper left',ncols=2)
save(fig,'validation_class_recalls','Same 80 validation cases. Adaptation: best saved checkpoint before overall timeout; reload not verified.')

fig,axes=plt.subplots(1,3,figsize=(19,6))
for ax,key,title in zip(axes,['accuracy','balanced_accuracy','loss'],['Accuracy (%)','Balanced accuracy (%)','Cross-entropy loss']):
 factor=1 if key=='loss' else 100
 for off,data,color,label in [(-.18,train,'#245A81','Training'),(.18,val,'#B95645','Validation')]:
  for i,k in enumerate(ids):
   value=data[k][key]
   if value:
    v=float(value)*factor;ax.bar(i+off,v,.33,color=color,label=label if i==0 else None)
   else: ax.text(i+off,0,'?',ha='center',va='bottom',fontsize=16,color=color)
 ax.set(xticks=x,xticklabels=['GRU\nAdam','GRU\nAdamW','Early\ngoal','Summary\nstatic','Grid\nstatic','Adapt\nblock'],title=title)
 if key!='loss':ax.set_ylim(0,105)
 ax.legend(fontsize=10)
fig.suptitle('Training and validation at each run’s selected checkpoint',fontsize=17)
save(fig,'selected_train_validation','? = value not supplied in reviewed records; not zero. Selected epochs: 53, 32, 52, 42, 18, 3.\nAdaptation stopped at overall time limit; final checks not saved. These are snapshots, not epoch curves.')
print('Generated three SVG/PNG chart pairs from selected_metrics.csv.')

# One selected validation confusion matrix per model, from reported counts.
names = dict(zip(ids, labels))
for row in csv.DictReader((ROOT / 'confusion_counts.csv').open()):
    if row['partition'] != 'validation':
        continue
    key = row['run']
    counts = np.array([[int(row['true_retract_pred_retract']), int(row['true_retract_pred_advance'])],
                       [int(row['true_advance_pred_retract']), int(row['true_advance_pred_advance'])]])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(counts, cmap='Blues', vmin=0, vmax=46)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(counts[i,j]), ha='center', va='center', fontsize=26,
                    color='white' if counts[i,j]>25 else '#16364a')
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['RETRACT','ADVANCE'],
           yticklabels=['RETRACT','ADVANCE'], xlabel='Predicted action', ylabel='True action',
           title=names[key].replace('\n',' — '))
    caption = f"Validation · 80 cases · selected epoch {row['selected_epoch']}. Rows=true; columns=predicted."
    if key == 'lastblock-static':
        caption += '\nBest saved checkpoint before overall timeout; reload not verified.'
    save(fig, 'confusion_'+key, caption)
print('Saved six per-model validation confusion matrices in SVG and PNG.')
