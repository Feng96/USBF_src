"""Tables and figures computed from the freshly rescored fixed geometries."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
from .classification import features,classify
from usbf.io.common import save

CLASSES=['unchanged','circle_ellipse_fit','hole_geometry_change','component_topology_change',
         'long_axis_change','net_area_expansion','net_area_contraction','near_constant_area_edit']

def build(data,original,product,registrations,summary,output):
    from .replay import summarize
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ga=original.to_crs(32611);us=product.to_crs(32611);records=[]
    for row in data.itertuples():
        state=us.loc[row.key]
        m=SimpleNamespace(source=row.source,parent=row.parent,patch_id=row.patch_id,modified=row.gated_modified,
            circle_excluded=bool(row.original_circle and row.gated_modified),pipeline_hollow=bool(state.get('hollow',False)),
            pipeline_small_FP=bool(state.get('possible_small_FP',False)),pipeline_decision=str(state.get('V7_decision','')))
        records.append(features(row.key,ga.loc[row.key].geometry,us.loc[row.key].geometry,m))
    f=pd.DataFrame(records).set_index('key');q=data.join(f.drop(columns=[c for c in f if c in data]))
    q.to_csv(output/'GEOMETRY_CLASSIFICATION.csv',index=False)
    classes={c:q[q.gated_modified&q.primary_class.eq(c)] for c in CLASSES if c!='unchanged'}
    s4=summarize(q,classes);s4.to_csv(output/'TableS4_geometry_classes.csv',index=False,na_rep='NA')
    s4[['population','reference_frame','metric','objects','improved','stable','worsened']].to_csv(output/'TableS5_outcomes.csv',index=False)
    s6=summary.copy();s6['comparator']='GBA'
    fixed=q[q.original_modified].copy()
    for frame in ('raw_reference','reference_in_NAIP_frame'):
        for metric in ('iou','bf1_1m','bf1_2m'):fixed[frame+'_GBA_'+metric]=fixed[frame+'_USBF_old_'+metric]
    diff=summarize(fixed,{'fixed180_before_vs_after_confirmation':fixed});diff['comparator']='USBF_old'
    pd.concat([s6,diff],ignore_index=True).to_csv(output/'TableS6_frame_and_policy_sensitivity.csv',index=False,na_rep='NA')
    strata={}
    for column in ('size_bin','shape_bin','source','elongation_bin','centroid_shift_bin','orientation_bin','net_area_bin'):
        for value,r in q[q.gated_modified].groupby(column):strata[column+':'+str(value)]=r
    s7=summarize(q,strata);s7.to_csv(output/'TableS7_contextual_strata.csv',index=False,na_rep='NA')
    sensitivity=[]
    modified=q[q.gated_modified].copy()
    for angle in (3,5,10):
        for area in (.03,.05,.10):
            names=modified.apply(lambda r:classify(r.to_dict(),angle,area),axis=1)
            groups={c:modified[names.eq(c)] for c in CLASSES if c!='unchanged'}
            table=summarize(q,groups);table['angle_threshold_deg']=angle;table['net_area_threshold']=area;sensitivity.append(table)
    s8=pd.concat(sensitivity,ignore_index=True);s8.to_csv(output/'CLASSIFICATION_THRESHOLD_SENSITIVITY.csv',index=False,na_rep='NA')
    matched=q.groupby('parent').agg(matched_buildings=('key','size'),scored_contexts=('patch_id','nunique'),modified=('gated_modified','sum'))
    accepted=registrations[registrations.status.eq('accepted')].groupby('parent').size()
    matched['accepted_registration_contexts']=accepted;matched.to_csv(output/'TableS9_parent_image_counts.csv')
    save(output/'SUPPLEMENT_COUNTS.json',dict(fixed_pairs=len(q),modified=int(q.gated_modified.sum()),
        unchanged=int((~q.gated_modified).sum()),geometry_classes=modified.primary_class.value_counts().to_dict(),
        accepted_circle_replacements=int((modified.primary_class=='circle_ellipse_fit').sum()),
        accepted_hole_changes=int((modified.primary_class=='hole_geometry_change').sum()),
        bootstrap_replicates=5000,bootstrap_seed=20261024,empty_group_mean='NA',
        original_and_registered_reference_frames_preserved=True))
    plot(q,summary,s4,s7,s8,output)

def plot(data,summary,classes,strata,sensitivity,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def savefig(fig,name):
        fig.savefig(output/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(output/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    def comparison(table,codes,labels,name):
        fig,axs=plt.subplots(1,2,figsize=(10,4.5))
        for ax,metric in zip(axs,('iou','bf1_1m')):
            for frame,off,color,title in [('raw_reference',-.12,'#4477aa','Original reference coordinates'),('reference_in_NAIP_frame',.12,'#dd9933','Reference registered to NAIP')]:
                for j,c in enumerate(codes):
                    x=table[table.population.eq(c)&table.metric.eq(metric)&table.reference_frame.eq(frame)&table['product'].eq('USBF_OSM_gate')]
                    if len(x)!=1 or x.iloc[0].objects==0:continue
                    r=x.iloc[0];ax.plot(r.delta_pp,j+off,'o',color=color,mfc='white' if r.objects<10 else color,label=title if j==0 else None)
                    if np.isfinite(r.ci_low_pp):ax.hlines(j+off,r.ci_low_pp,r.ci_high_pp,color=color)
            ax.axvline(0,c='.5',ls='--');ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
            ax.set_xlabel('USBF − GBA (percentage points)');ax.set_title('IoU' if metric=='iou' else 'Boundary F1 (1 m)');ax.grid(axis='x',alpha=.15)
        h,l=axs[0].get_legend_handles_labels()
        if h:fig.legend(h,l,loc='lower center',ncol=2)
        fig.tight_layout(rect=[0,.1,1,1]);savefig(fig,name)
    comparison(summary,['addition_only','removal_only','addition_and_removal'],['Additions only','Removals only','Additions and removals'],'Figure3')
    fig,axs=plt.subplots(1,2,figsize=(10,4));circle=data.gated_modified&data.primary_class.eq('circle_ellipse_fit')
    nums=[int((~data.gated_modified).sum()),int((data.gated_modified&~circle).sum()),int(circle.sum())]
    axs[0].barh(range(3),nums,color=['#abb6bf','#4477aa','#dd9933']);axs[0].set_yticks(range(3),['Unchanged','Local revisions','Accepted circle/ellipse']);axs[0].invert_yaxis()
    for i,n in enumerate(nums):axs[0].text(n+10,i,str(n),va='center')
    counts=data[data.gated_modified].primary_class.value_counts();axs[1].barh(range(len(counts)),counts.values);axs[1].set_yticks(range(len(counts)),[c.replace('_',' ') for c in counts.index]);axs[1].invert_yaxis()
    fig.tight_layout();savefig(fig,'FigureS1')
    codes=[c for c in CLASSES[1:] if (data[data.gated_modified].primary_class==c).any()]
    comparison(classes,codes,[c.replace('_',' ') for c in codes],'FigureS2')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axs,('iou','bf1_1m')):
        left=np.zeros(len(codes))
        for state,color in [('improved','#228833'),('stable','#bbbbbb'),('worsened','#cc6677')]:
            values=[int(classes[classes.population.eq(c)&classes.metric.eq(metric)&classes.reference_frame.eq('reference_in_NAIP_frame')].iloc[0][state]) for c in codes]
            ax.barh(range(len(codes)),values,left=left,color=color,label=state);left+=values
        ax.set_yticks(range(len(codes)),[c.replace('_',' ') for c in codes]);ax.invert_yaxis();ax.set_title(metric);ax.legend()
    fig.tight_layout();savefig(fig,'FigureS3')
    codes=[c for c in strata.population.unique() if c.startswith(('size_bin:','shape_bin:','source:'))]
    comparison(strata,codes,[c.replace('_',' ') for c in codes],'FigureS4')
    fig,axs=plt.subplots(2,2,figsize=(9,6))
    for i,frame in enumerate(('raw_reference','reference_in_NAIP_frame')):
        for j,metric in enumerate(('iou','bf1_1m')):
            ss=summary[summary.population.eq('fixed180')&summary.reference_frame.eq(frame)&summary.metric.eq(metric)]
            a=ss[ss['product'].eq('USBF_old')].iloc[0];b=ss[ss['product'].eq('USBF_OSM_gate')].iloc[0]
            vals=[100*a.gba_mean,100*a.usbf_mean,100*b.usbf_mean];ax=axs[i,j]
            ax.bar(range(3),vals,color=['#778899','#dd9933','#228833']);ax.set_ylim(0,100);ax.set_xticks(range(3),['GBA','Before confirmation','After confirmation']);ax.set_title(frame+' / '+metric,fontsize=8)
            for k,v in enumerate(vals):ax.text(k,v+1,f'{v:.2f}',ha='center')
    fig.tight_layout();savefig(fig,'FigureS5')
    fig,axs=plt.subplots(2,4,figsize=(12,6));axes=list(axs.flat)
    for ax,c in zip(axes,CLASSES[1:]):
        t=sensitivity[sensitivity.population.eq(c)&sensitivity.metric.eq('iou')&sensitivity.reference_frame.eq('reference_in_NAIP_frame')]
        values=t.pivot(index='angle_threshold_deg',columns='net_area_threshold',values='delta_pp').reindex(index=[3,5,10],columns=[.03,.05,.10])
        counts=t.pivot(index='angle_threshold_deg',columns='net_area_threshold',values='objects').reindex(index=[3,5,10],columns=[.03,.05,.10])
        ax.imshow(np.ma.masked_invalid(values.to_numpy()),cmap='RdYlGn',vmin=-10,vmax=10)
        for y in range(3):
            for x in range(3):
                n=int(counts.iloc[y,x]);text='n=0\nNA' if n==0 else f'n={n}\n{values.iloc[y,x]:+.2f}'
                ax.text(x,y,text,ha='center',va='center',fontsize=7)
        ax.set_xticks(range(3),['3%','5%','10%']);ax.set_yticks(range(3),['3°','5°','10°']);ax.set_title(c.replace('_',' '),fontsize=8)
    for ax in axes[len(CLASSES)-1:]:ax.set_visible(False)
    fig.tight_layout();savefig(fig,'FigureS6')
