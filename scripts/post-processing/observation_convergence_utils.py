"""Descriptive parameter restriction on unchanged saved observation scores."""
import json
import math
import numpy as np
import pandas as pd
from run_summary_utils import SUBSETS
from parameter_labels import parameter_label


def readable_model_labels(evaluations):
    """Snapshot-local labels; fingerprints remain the persistent exact identifiers."""
    data=evaluations.copy()
    if 'evaluation_id' not in data:
        raise ValueError('Each saved model needs a unique evaluation_id.')
    if data.evaluation_id.duplicated().any():
        raise ValueError('Duplicate evaluation_id values; supply unique saved evaluation identifiers.')
    names={key:f'Model {i:04d}' for i,key in enumerate(sorted(data.evaluation_id.unique()),1)}
    data['model_id']=data.evaluation_id.map(names)
    pool_keys=data.comparison_key.where(data.known_physical_metadata,
                                       data.comparison_key+':'+data.history)
    pools={key:f'Pool {i:02d}' for i,key in enumerate(sorted(pool_keys.unique()),1)}
    data['pool_id']=pool_keys.map(pools)
    return data


def model_pool(evaluations):
    """Preserve every saved evaluation, including identical parameter sets."""
    return evaluations.copy().reset_index(drop=True).assign(realisations=1)


def objective_columns(name):
    """Resolve readable product labels without mutating the shared SUBSETS registry."""
    if name in SUBSETS:return list(SUBSETS[name])
    from run_summary_utils import INTERFEROMETRY
    columns=[]
    for part in name.split('+'):
        if part=='SED':columns.append('score_sed')
        elif part=='ALMA':columns.append('score_alma')
        elif part in {'INT(band)','INT(chrom)'}:columns.extend(INTERFEROMETRY)
        elif part.startswith('PDI(') and part.endswith(')'):
            bands=part[4:-1].split(',')
            if not bands or any(b not in 'VIH' or len(b)!=1 for b in bands):
                raise ValueError('Unknown PDI bands: '+name)
            columns.extend('score_pdi_'+b for b in bands)
        else:raise ValueError('Unknown observation objective: '+name)
    return list(dict.fromkeys(columns))


def select_models(pool, subset, mode, value, absolute_allowance=0.):
    if not np.isfinite(value) or value < 0 or not np.isfinite(absolute_allowance) or absolute_allowance < 0:
        raise ValueError('Selection allowances must be finite and nonnegative.')
    columns=objective_columns(subset)
    scores = pool[columns].sum(axis=1, min_count=len(columns))
    if not np.isfinite(scores).all():
        raise ValueError('Every candidate must have finite scores for the objective.')
    if mode == 'allowance':
        cutoff = scores.min()+max(abs(scores.min())*value, absolute_allowance)
        keep = scores <= cutoff
    elif mode == 'fraction':
        if not 0 < value <= 1:
            raise ValueError('Retained fraction must be in (0,1].')
        # Include all ties at the cutoff; report realised counts explicitly.
        cutoff = scores.nsmallest(max(1, math.ceil(len(scores)*value))).max()
        keep = scores <= cutoff
    else:
        raise ValueError('Unknown selection mode.')
    return pool.loc[keep].copy(), float(cutoff)


def parameter_reference(pool, name, definition, period=None, bins=15):
    """Fixed type, coordinates and bins for all scopes, weightings and prefixes."""
    raw=pool['param__'+name].dropna()
    numeric=pd.to_numeric(raw,errors='coerce')
    declared_numeric=definition.get('type','').startswith(('uniform_', 'normal_', 'beta_'))
    categorical=definition.get('type') in {'categorical','ordinal'} or (not declared_numeric and numeric.isna().any())
    if categorical:
        choices=definition.get('choices',definition.get('sequence',[]))
        # Retain declared categories even when the saved pool never visited them.
        categories=list(dict.fromkeys([str(v) for v in choices]+sorted(raw.map(str).unique())))
        return dict(kind='categorical',categories=categories)
    full=numeric.to_numpy(float)
    full=full[np.isfinite(full)]
    lower,upper=definition.get('lower'),definition.get('upper')
    lower=float(lower) if lower is not None else np.nan
    upper=float(upper) if upper is not None else np.nan
    origin=np.nan; resultant=np.nan
    if period is not None:
        if not np.isfinite(period) or period<=0:raise ValueError('Angular period must be positive.')
        vector=np.exp(2j*np.pi*full/period).mean() if len(full) else complex(np.nan,np.nan)
        resultant=abs(vector)
        origin=float(np.angle(vector)*period/(2*np.pi)) if np.isfinite(resultant) and resultant>1e-12 else 0.
        full=(full-origin+period/2)%period-period/2
        lower,upper=-period/2,period/2;kind='circular'
    elif definition.get('log'):
        full=np.log10(full[full>0])
        lower=np.log10(lower) if lower>0 else np.nan
        upper=np.log10(upper) if upper>0 else np.nan
        kind='log10'
    else:kind='linear'
    left=lower if np.isfinite(lower) else (full.min() if len(full) else 0.)
    right=upper if np.isfinite(upper) else (full.max() if len(full) else 1.)
    if len(full):left=min(left,full.min());right=max(right,full.max())
    if left==right:left-=.5;right+=.5
    return dict(kind=kind,lower=lower,upper=upper,origin=origin,period=period,
                reference_resultant=resultant,edges=np.linspace(left,right,bins+1).tolist())


def parameter_values(frame, name, reference):
    """Return valid plotting/metric values in the reference's fixed coordinates."""
    raw=frame['param__'+name]
    if reference['kind']=='categorical':return raw.dropna().map(str)
    values=pd.to_numeric(raw,errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
    if reference['kind']=='log10':return np.log10(values[values>0])
    if reference['kind']=='circular':
        return (values-reference['origin']+reference['period']/2)%reference['period']-reference['period']/2
    return values


def parameter_metrics(pool, selected, name, definition, min_models=20, period=None, bins=15, reference=None):
    """Descriptive metrics using shared bins/origin and only valid parameter values."""
    ref=reference if reference is not None else parameter_reference(pool,name,definition,period,bins)
    sample=parameter_values(selected,name,ref)
    full=parameter_values(pool,name,ref)
    invalid=int(selected['param__'+name].notna().sum()-len(sample))
    result=dict(parameter=name,kind=ref['kind'],n_pool=len(full),n_selected=len(sample),
                n_missing=int(selected['param__'+name].isna().sum()),n_invalid=invalid,
                supported=len(sample)>=min_models and invalid==0)
    if ref['kind']=='categorical':
        frequencies=sample.value_counts(normalize=True).reindex(ref['categories'],fill_value=0)
        result.update(retained_categories=int((frequencies>0).sum()),
                      category_fractions=json.dumps(frequencies.to_dict()))
        return result,frequencies.to_numpy(),ref['categories']
    # Report actual saved values separately from transformed width coordinates.
    # Quantiles are interpolated in saved units, not exponentiated log quantiles.
    physical_values = pd.to_numeric(selected.loc[sample.index, 'param__'+name])
    for label, quantile in [('minimum', 0.), ('q10', .1), ('q25', .25),
                            ('median', .5), ('q75', .75), ('q90', .9), ('maximum', 1.)]:
        result[label+'_in_parameter_units'] = physical_values.quantile(quantile)
    values=sample.to_numpy(float)
    q10,median,q90=np.quantile(values,[.1,.5,.9]) if len(values) else [np.nan]*3
    lower,upper=ref['lower'],ref['upper'];span=upper-lower
    hist=np.histogram(values,ref['edges'])[0].astype(float)
    if hist.sum():hist/=hist.sum()
    result.update(q10=q10,median=median,q90=q90,width=q90-q10,
        width_fraction_of_bounds=(q90-q10)/span if span>0 else np.nan,
        median_in_parameter_units=result['median_in_parameter_units'],circular_origin=ref['origin'],
        reference_circular_resultant=ref['reference_resultant'],
        circular_resultant=abs(np.exp(2j*np.pi*values/ref['period']).mean()) if len(values) and ref['kind']=='circular' else np.nan,
        fraction_near_boundary=float(((values<=lower+.02*span)|(values>=upper-.02*span)).mean())
            if span>0 and len(values) and ref['kind']!='circular' else np.nan,
        fraction_outside_bounds=float(((values<lower)|(values>upper)).mean())
            if span>0 and len(values) and ref['kind']!='circular' else np.nan,
        histogram=json.dumps(hist.tolist()),histogram_edges=json.dumps(ref['edges']))
    return result,hist,ref['edges']


def restriction_analysis(pool, definitions, sequence, parameters, allowances, fractions,
                         absolute_allowance=0., min_models=20, periods=None, bins=15, references=None):
    if len(sequence) < 2:
        raise ValueError('Provide at least two named observation subsets.')
    for old,new in zip(sequence, sequence[1:]):
        if not set(objective_columns(old)) < set(objective_columns(new)):
            raise ValueError('Each addition must strictly retain and expand the previous terms.')
    if pool.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows, selections, contrasts = [], [], []
    periods = periods or {}
    for mode, levels in [('allowance', allowances), ('fraction', fractions)]:
        for level in sorted(set(levels)):
            previous = {}
            for step,subset in enumerate(sequence):
                selected, cutoff = select_models(pool,subset,mode,level,absolute_allowance)
                meta = dict(mode=mode, level=level, step=step, subset=subset)
                selections.append(dict(meta, n_pool=len(pool), n_selected=len(selected),
                                       retained_fraction=len(selected)/len(pool), cutoff=cutoff,
                                       supported=len(selected)>=min_models))
                for name in parameters:
                    result,hist,edges = parameter_metrics(pool,selected,name,definitions.get(name,{}),
                                                          min_models, periods.get(name), bins, (references or {}).get(name))
                    # Retain all tied best values; never imply that the sample median
                    # is an evaluated model or that one tie was uniquely preferred.
                    objective_scores = selected[objective_columns(subset)].sum(axis=1)
                    best = selected.loc[objective_scores.eq(objective_scores.min()), 'param__'+name]
                    result['best_model_parameter_values'] = json.dumps(best.dropna().unique().tolist())
                    rows.append(dict(meta,**result))
                    if name in previous:
                        old,old_hist = previous[name]
                        supported = bool(result['supported'] and old['supported'])
                        change = dict(meta,parameter=name,from_subset=sequence[step-1],
                                      supported=supported,kind=result['kind'],
                                      distribution_overlap=float(np.minimum(hist,old_hist).sum())
                                          if len(hist) and len(hist)==len(old_hist) else np.nan)
                        if 'width' in result and 'width' in old:
                            shift = result['median']-old['median']
                            if name in periods: shift=(shift+periods[name]/2)%periods[name]-periods[name]/2
                            change.update(width_ratio=result['width']/old['width'] if old['width']>0 else np.nan,
                                          median_shift=shift,
                                          shift_in_previous_width=shift/old['width'] if old['width']>0 else np.nan)
                        contrasts.append(change)
                    previous[name]=(result,hist)
    return pd.DataFrame(rows),pd.DataFrame(selections),pd.DataFrame(contrasts)


def plot_restriction(metrics, parameters, mode='allowance', value=0.1):
    import matplotlib.pyplot as plt
    data=metrics[(metrics['mode']==mode) & np.isclose(metrics.level,value)]
    names=[n for n in parameters if n in data.parameter.values]
    if not names: return None
    ncols = 3
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(15, 3.5 * nrows),
        squeeze=False)
    # fig,axes=plt.subplots(math.ceil(len(names)/2),2,figsize=(12,3.3*math.ceil(len(names)/2)),squeeze=False)
    for ax,name in zip(axes.flat,names):
        rows=data[data.parameter==name].sort_values('step')
        if rows.iloc[0]['kind']=='categorical':
            for category in json.loads(rows.iloc[0].category_fractions):
                ax.plot(rows.step,[json.loads(v).get(category,0) for v in rows.category_fractions], 'o-',label=category)
            ax.set_ylabel('Selected category fraction');ax.legend(fontsize=7)
        elif 'median' in rows:
            # Endpoints are in saved physical units; logarithmic axes preserve
            # readable mass values while displaying a wide dynamic range.
            ax.vlines(rows.step, rows.q10_in_parameter_units, rows.q90_in_parameter_units,
                      color='#2166ac', lw=2, label='10–90% range')
            ax.vlines(rows.step, rows.q25_in_parameter_units, rows.q75_in_parameter_units,
                      color='#2166ac', lw=6, label='25–75% range')
            ax.scatter(rows.step, rows.median_in_parameter_units,
                       facecolors='none', edgecolors='#2166ac', label='Median')
            good=rows[rows.supported]
            ax.scatter(good.step, good.median_in_parameter_units, color='#2166ac')
            if rows.iloc[0]['kind']=='log10': ax.set_yscale('log')
            ax.set_ylabel('Parameter value (saved units)')
            ax.legend(fontsize=7)
        ax.set_xticks(rows.step,rows.subset,rotation=20,ha='right')
        ax.set_title(parameter_label(name)+(' — insufficient models' if not rows.supported.all() else ''))
    for ax in axes.flat[len(names):]: ax.set_visible(False)
    fig.suptitle(f'Parameter restriction: {mode}={value} (sampled models, not confidence intervals)')
    fig.tight_layout();return fig


def plot_sensitivity(changes, parameters):
    import matplotlib.pyplot as plt
    names=[n for n in parameters if n in changes.parameter.values]
    if not names:return None
    fig,axes=plt.subplots(math.ceil(len(names)/2),2,figsize=(12,3.3*math.ceil(len(names)/2)),squeeze=False)
    for ax,name in zip(axes.flat,names):
        data=changes[(changes.parameter==name)&(changes['mode']=='allowance')]
        for addition,rows in data.groupby('subset',sort=False):
            rows=rows.sort_values('level')
            if rows.iloc[0]['kind']=='categorical':
                ax.plot(rows.level,rows.distribution_overlap.where(rows.supported),'o-',label=addition)
            elif 'width_ratio' in rows:
                ax.plot(rows.level,rows.width_ratio.where(rows.supported),'o-',label=addition)
        ax.axhline(1,color='black',ls='--',lw=1)
        ax.set(title=parameter_label(name),xlabel='Relative allowance',ylabel=('Category distribution overlap' if len(data) and data.iloc[0]['kind']=='categorical' else 'New / previous 10–90% width'))
        if ax.lines and len(ax.lines)>1:ax.legend(fontsize=7)
    for ax in axes.flat[len(names):]:ax.set_visible(False)
    fig.tight_layout();return fig


def fidelity_sequences(evaluations, records, source='recorded_stages'):
    """Discover observation additions, preserving actual products and provenance.

    recorded_stages follows budget order within each history. observed_products
    builds maximal nested comparisons from observed sets across histories; those
    are rescoring comparisons and do not claim a historical execution order.
    """
    from run_summary_utils import products_for, objective_label, INTERFEROMETRY
    mapping={'sed':['score_sed'],'alma':['score_alma'],
             'vis2_chromatic':INTERFEROMETRY,'vis2_1perband':INTERFEROMETRY,
             **{'pdi_'+b:['score_pdi_'+b] for b in 'VIH'}}
    rows=[]
    for _,row in evaluations.loc[evaluations.usable].iterrows():
        products=products_for(records[row.evaluation_id]['info'])
        unknown=set(products)-set(mapping)
        if unknown or not products:
            continue
        name=objective_label(products)
        rows.append(dict(history=row.history,budget=row.budget,stage=row.stage,
                         objective=name,products=', '.join(products),
                         product_source='recorded' if records[row.evaluation_id]['info'].get('fidelity',{}).get('products') else 'inferred',
                         product_set=frozenset(products)))
    stages=pd.DataFrame(rows)
    if stages.empty:
        return {},pd.DataFrame(),pd.DataFrame()
    stages=stages.drop_duplicates(['history','budget','stage','objective','product_source'])
    paths={}
    if source=='recorded_stages':
        for history,group in stages.groupby('history',sort=False):
            group=group.sort_values('budget')
            # Ambiguous observation sets at one budget must not acquire invented order.
            if group.groupby('budget').objective.nunique().gt(1).any():
                continue
            path=[];previous=frozenset();previous_columns=set()
            for row in group.itertuples():
                if path and row.objective==path[-1]:continue
                if path and (not previous < row.product_set or not previous_columns < set(objective_columns(row.objective))):
                    if len(path)>1:paths.setdefault(tuple(path),set()).add(history)
                    path=[]
                path.append(row.objective);previous=row.product_set;previous_columns=set(objective_columns(row.objective))
            if len(path)>1:paths.setdefault(tuple(path),set()).add(history)
    elif source=='observed_products':
        objectives=stages.drop_duplicates('objective').set_index('objective').product_set.to_dict()
        def expands(a,b):
            return objectives[a] < objectives[b] and set(objective_columns(a)) < set(objective_columns(b))
        def extend(path):
            successors=[b for b in objectives if expands(path[-1],b)]
            # Only adjacent observed sets: do not generate all skipped-stage paths.
            successors=[b for b in successors if not any(expands(path[-1],c) and expands(c,b) for c in successors)]
            if not successors and len(path)>1:
                paths[tuple(path)]=set(stages.loc[stages.objective.isin(path),'history'])
            for b in successors:extend(path+[b])
        for a in objectives:
            if not any(expands(b,a) for b in objectives):extend([a])
    else:
        raise ValueError('Sequence source must be recorded_stages or observed_products.')
    sequences={' → '.join(path):list(path) for path in paths}
    provenance=pd.DataFrame([dict(sequence=' → '.join(path),source=source,
        histories=' | '.join(sorted(histories))) for path,histories in paths.items()])
    return sequences,stages.drop(columns='product_set'),provenance


def pool_score_summary(evaluations, observables=('SED','interferometry','PDI','ALMA'),
                       parameter_filters=None, parameter_hashes=None, rtol=1e-6, atol=0., model_ids=None):
    """One row per saved evaluation; no merging or averaging of parameter sets.

    Missing requested contributions make the selected total undefined.
    """
    from run_summary_utils import SCORE_COLUMNS, INTERFEROMETRY, PDI
    definitions={**{name:[column] for name,column in SCORE_COLUMNS.items()},
                 'interferometry':INTERFEROMETRY,'PDI':PDI,'All':SUBSETS['All']}
    if not observables or any(name not in definitions for name in observables):
        raise ValueError('Select observable names from SCORE_COLUMNS or interferometry, PDI, All.')
    terms=list(dict.fromkeys(column for name in observables for column in definitions[name]))
    labeled = evaluations if {'model_id','pool_id'}.issubset(evaluations.columns) else readable_model_labels(evaluations)
    data=labeled.loc[labeled.usable].copy()
    if model_ids is not None:
        data=data[data.model_id.isin(model_ids)]
    if parameter_hashes is not None:
        data=data[data.parameter_hash.isin(parameter_hashes)]
    for name,criterion in (parameter_filters or {}).items():
        column='param__'+name
        if column not in data:raise ValueError('Unknown parameter: '+name)
        if isinstance(criterion,dict):
            if not set(criterion)<= {'min','max'} or not criterion:
                raise ValueError('Range filters require min and/or max.')
            numeric=pd.to_numeric(data[column],errors='coerce')
            if 'min' in criterion:data=data.loc[numeric>=criterion['min']];numeric=numeric.reindex(data.index)
            if 'max' in criterion:data=data.loc[numeric<=criterion['max']]
        elif isinstance(criterion,(list,tuple,set)):
            data=data[data[column].isin(criterion)]
        elif isinstance(criterion,(int,float,np.number)):
            values=pd.to_numeric(data[column],errors='coerce')
            data=data[np.isclose(values,criterion,rtol=rtol,atol=atol,equal_nan=False)]
        else:
            data=data[data[column]==criterion]
    # Missing physical provenance must never silently combine different histories.
    data['score_pool']=data.comparison_key.where(data.known_physical_metadata,
        data.comparison_key+':'+data.history)
    data['selected_total']=data[terms].sum(axis=1,min_count=len(terms))
    data.loc[~np.isfinite(data[terms]).all(axis=1),'selected_total']=np.nan
    groups={'score_interferometry_total':INTERFEROMETRY,'score_PDI_total':PDI,'score_all_total':SUBSETS['All']}
    for name,columns in groups.items():
        data[name]=data[columns].sum(axis=1,min_count=len(columns))
        data.loc[~np.isfinite(data[columns]).all(axis=1),name]=np.nan
    data['selected_observables']=' + '.join(observables)
    data['selected_total_complete']=data.selected_total.notna()
    summary_columns=['model_id','evaluation_id','pool_id','score_pool','comparison_key',
                     'history','config_id','budget','stage','objective_key','products','parameter_hash',
                     'selected_observables','selected_total_complete','selected_total','cost',
                     *SCORE_COLUMNS.values(),*groups,
                     *[c for c in data if c.startswith('param__')]]
    return data.reindex(columns=list(dict.fromkeys(summary_columns))),data


def convergence_views(evaluations):
    """Return evaluation and equal-configuration pools plus repeat-score diagnostics.

    Group only exact parameter fingerprints with known matching physical/scoring/
    numerical metadata, budget and instance. Missing physical provenance leaves
    evaluations separate. Original evaluation IDs and score lookup are untouched.
    """
    data=evaluations.copy().reset_index(drop=True)
    required=['evaluation_id','parameter_hash','comparison_key','known_physical_metadata','budget']
    if any(c not in data for c in required):
        raise ValueError('Convergence weighting requires saved evaluation and compatibility metadata.')
    if data.evaluation_id.duplicated().any():
        raise ValueError('Evaluation IDs must be unique.')
    data['_instance']=data['instance'] if 'instance' in data else 'unknown'
    data['_compatibility']=data.comparison_key.where(data.known_physical_metadata,
                                                     data.comparison_key+':'+data.evaluation_id)
    keys=['_compatibility','budget','_instance','parameter_hash']
    rows,scatter=[],[]
    for index,(_,group) in enumerate(data.groupby(keys,dropna=False,sort=True),1):
        label=f'Configuration {index:04d}'
        columns=SUBSETS['All']
        if not np.isfinite(group[columns]).all().all():
            raise ValueError('Both convergence views require complete finite score vectors.')
        row={c:group[c].iloc[0] for c in group if c.startswith('param__')}
        row.update(configuration_id=label,parameter_hash=group.parameter_hash.iloc[0],
                   comparison_key=group.comparison_key.iloc[0],budget=group.budget.iloc[0],
                   instance=group['_instance'].iloc[0],realisations=len(group),
                   source_evaluations=json.dumps(group.evaluation_id.tolist()),
                   source_model_ids=json.dumps(group.model_id.tolist()) if 'model_id' in group else '[]',
                   source_histories=' | '.join(sorted(group.history.unique())) if 'history' in group else '')
        row.update(group[columns].mean().to_dict())
        rows.append(row)
        data.loc[group.index,'configuration_id']=label
        diagnostics={c:group[c] for c in columns}
        diagnostics['all_score_total']=group[columns].sum(axis=1,min_count=len(columns))
        for name,values in diagnostics.items():
            scatter.append(dict(configuration_id=label,parameter_hash=row['parameter_hash'],
                comparison_key=row['comparison_key'],budget=row['budget'],score=name,
                n_evaluations=len(group),n_simulation_seeds=group.seed.nunique(dropna=True) if 'seed' in group else np.nan,
                mean=values.mean(),std=values.std(ddof=1),minimum=values.min(),maximum=values.max(),
                source_evaluations=row['source_evaluations']))
    data=data.drop(columns=['_instance','_compatibility'])
    return {'configuration':pd.DataFrame(rows),'evaluation':model_pool(data)},pd.DataFrame(scatter)


def plot_weighting_comparison(metrics, parameters, mode='allowance', level=.1):
    """Compare widths using both views without pooling their rows."""
    import matplotlib.pyplot as plt
    data=metrics[(metrics['mode']==mode)&np.isclose(metrics.level,level)]
    names=[n for n in parameters if n in data.parameter.values]
    if not names:return None
    fig,axes=plt.subplots(math.ceil(len(names)/2),2,figsize=(12,3.3*math.ceil(len(names)/2)),squeeze=False)
    for ax,name in zip(axes.flat,names):
        for weighting,rows in data[data.parameter==name].groupby('weighting',sort=False):
            rows=rows.sort_values('step')
            if rows.iloc[0]['kind']=='categorical':
                ax.plot(rows.step,rows.retained_categories.where(rows.supported),'o-',label=weighting)
            elif 'width_fraction_of_bounds' in rows:
                ax.plot(rows.step,rows.width_fraction_of_bounds.where(rows.supported),'o-',label=weighting)
            ax.set_xticks(rows.step,rows.subset,rotation=20,ha='right')
        ax.set(title=parameter_label(name),ylabel=('Retained categories' if data[data.parameter==name].iloc[0]['kind']=='categorical' else '10–90% width / allowed interval'))
        if ax.lines:ax.legend(fontsize=7)
    for ax in axes.flat[len(names):]:ax.set_visible(False)
    fig.suptitle('Weighting sensitivity: unsupported samples masked; categories use fraction tables')
    fig.tight_layout();return fig


def compatible_sequence_paths(sequences, stages, selected, records):
    """Only rescore labels whose actual products exist in the selected pool."""
    from run_summary_utils import products_for
    available=set.intersection(*(set(products_for(records[e]['info'])) for e in selected.evaluation_id))
    definitions={row.objective:set(row.products.split(', ')) for row in stages.itertuples()}
    accepted={};audit=[]
    for name,path in sequences.items():
        missing=sorted(set().union(*(definitions[label]-available for label in path)))
        audit.append(dict(sequence=name,compatible=not missing,missing_products=', '.join(missing)))
        if not missing:accepted[name]=path
    return accepted,pd.DataFrame(audit)


def plot_parameter_marginals(pool, selections, names, references, min_models=20):
    """Continuous histograms and categorical bars on the same metric reference."""
    import matplotlib.pyplot as plt
    if not names:return None
    fig,axes=plt.subplots(math.ceil(len(names)/2),2,figsize=(12,3.8*math.ceil(len(names)/2)),squeeze=False)
    colors={label:f'C{i%10}' for i,label in enumerate(selections)}
    for ax,name in zip(axes.flat,names):
        ref=references[name]
        if ref['kind']=='categorical':
            categories=ref['categories'];x=np.arange(len(categories));width=.8/max(1,len(selections))
            for i,(label,sample) in enumerate(selections.items()):
                values=parameter_values(sample,name,ref)
                freq=values.value_counts(normalize=True).reindex(categories,fill_value=0)
                ax.bar(x+(i-(len(selections)-1)/2)*width,freq,width=width,
                       color=colors[label],alpha=.75,label=f'{label}: n={len(values)}'+(' (small sample)' if len(values)<min_models else ''))
            ax.set_xticks(x,categories,rotation=35,ha='right')
            ax.set(xlabel=parameter_label(name),ylabel='Fraction per allowed category',title='Discrete choices: category fractions')
        else:
            for label,sample in [('Common pool',pool),*selections.items()]:
                values=parameter_values(sample,name,ref)
                if not len(values):continue
                ax.hist(values,bins=ref['edges'],weights=np.ones(len(values))/len(values),histtype='step',
                        color='black' if label=='Common pool' else colors[label],
                        linestyle='--' if label=='Common pool' else '-',
                        label=f'{label}: n={len(values)}'+(' (small sample)' if len(values)<min_models else ''))
            unit='log10' if ref['kind']=='log10' else f"angle offset from {ref['origin']:.1f}°" if ref['kind']=='circular' else 'saved units'
            ax.set(xlabel=f'{parameter_label(name)} ({unit})',ylabel='Fraction per shared bin',title='Numeric distribution: histogram')
        ax.legend(fontsize=6)
    for ax in axes.flat[len(names):]:ax.set_visible(False)
    fig.tight_layout();return fig


def plot_joint_parameters(pool, selections, x, y, references, point_size=16, point_alpha=.55):
    import matplotlib.pyplot as plt
    if any(references[name]['kind']=='categorical' for name in (x,y)):return None
    fig,ax=plt.subplots(figsize=(7,5))
    for i,(label,sample) in enumerate([('Common pool',pool),*selections.items()]):
        a=parameter_values(sample,x,references[x]);b=parameter_values(sample,y,references[y])
        paired=pd.concat([a.rename('x'),b.rename('y')],axis=1).dropna()
        ax.scatter(paired.x,paired.y,color='black' if i==0 else f'C{(i-1)%10}',alpha=point_alpha, marker='o', linewidths=0,
                   s=point_size,label=f'{label}: paired n={len(paired)}')
    for name,axis in [(x,'x'),(y,'y')]:
        ref=references[name]
        units='log10' if ref['kind']=='log10' else 'fixed angular offset' if ref['kind']=='circular' else 'saved units'
        getattr(ax,'set_'+axis+'label')(f'{parameter_label(name)} ({units})')
    ax.legend(fontsize=7);fig.tight_layout();return fig


def parameter_score_diagnostic(pool, name, reference, sequence, observables=('SED','interferometry','PDI','ALMA'),
                               allowance=.1, absolute_allowance=0., min_bin_models=10,
                               lower_quantile=.1, score_scale='linear', point_size=16, point_alpha=.55):
    """One parameter: individual scores, cumulative objectives, additions, distributions.

    Every panel retains the same candidates. Binned statistics describe saved
    scores conditional on the parameter bin, not reoptimised profile costs.
    Returns the figure, panel-selection audit and per-bin/category score table.
    """
    import matplotlib.pyplot as plt
    from textwrap import fill
    from run_summary_utils import SCORE_COLUMNS, INTERFEROMETRY, PDI
    if min_bin_models<2 or not 0<lower_quantile<.5:
        raise ValueError('Require min_bin_models>=2 and 0<lower_quantile<0.5.')
    if score_scale not in {'linear','log'}:raise ValueError('Score scale must be linear or log.')
    if not np.isfinite(allowance) or allowance<0 or not np.isfinite(absolute_allowance) or absolute_allowance<0:
        raise ValueError('Allowances must be finite and nonnegative.')
    if len(sequence)<2:raise ValueError('Choose an observation-addition sequence.')
    for old,new in zip(sequence,sequence[1:]):
        if not set(objective_columns(old))<set(objective_columns(new)):
            raise ValueError('Sequence must strictly add score terms.')
    observable_columns={**{k:[v] for k,v in SCORE_COLUMNS.items()},
                        'interferometry':INTERFEROMETRY,'PDI':PDI,'All':SUBSETS['All']}
    if not observables or any(k not in observable_columns for k in observables):
        raise ValueError('Unknown or empty observable selection.')
    # Reset local positions only. IDs and scores remain unchanged.
    candidates=pool.reset_index(drop=True)
    values=parameter_values(candidates,name,reference)
    categorical=reference['kind']=='categorical'
    if categorical:
        locations={label:i for i,label in enumerate(reference['categories'])}
        x=values.map(locations).astype(float)
    else:x=values.astype(float)

    def scores(columns):
        series=candidates[columns].sum(axis=1,min_count=len(columns))
        if not np.isfinite(series).all():raise ValueError('Diagnostic requires complete finite requested scores.')
        return series

    def selection(series):
        cutoff=float(series.min()+max(abs(series.min())*allowance,absolute_allowance))
        return series.le(cutoff),cutoff

    panels=[];selected={}
    for observable in dict.fromkeys(observables):
        score=scores(observable_columns[observable]);mask,cutoff=selection(score)
        panels.append(dict(kind='individual',label=observable,score=score,mask=mask,cutoff=cutoff,
                           columns=observable_columns[observable],selection_objective=observable))
    for objective in sequence:
        score=scores(objective_columns(objective));mask,cutoff=selection(score)
        selected[objective]=mask
        identical=next((panel for panel in panels if set(panel.get('columns',[]))==set(objective_columns(objective))),None)
        if identical is not None:
            identical['kind']='individual/cumulative'
            if identical['label']!=objective:identical['label']+=' / '+objective
        else:
            panels.append(dict(kind='cumulative',label=objective,score=score,mask=mask,cutoff=cutoff,
                               columns=objective_columns(objective),selection_objective=objective))
    for old,new in zip(sequence,sequence[1:]):
        columns=[c for c in objective_columns(new) if c not in objective_columns(old)]
        panels.append(dict(kind='added',label=f'{old} → {new}',score=scores(columns),mask=selected[new],
                           cutoff=selection(scores(objective_columns(new)))[1],selection_objective=new))
    fig,axes=plt.subplots(math.ceil((len(panels)+1)/3),3,
                         figsize=(17,4.1*math.ceil((len(panels)+1)/3)),squeeze=False)
    audits=[];summaries=[]
    for panel_index,(ax,panel) in enumerate(zip(axes.flat,panels)):
        y=panel['score'].reindex(x.index);chosen=panel['mask'].reindex(x.index)
        ax.scatter(x,y,marker='o',s=point_size,color='black',alpha=point_alpha,linewidths=0,label=f'All: n={len(x)}',zorder=1)
        ax.scatter(x[chosen],y[chosen],marker='o',s=point_size,color='#2166ac',alpha=point_alpha,linewidths=0,
                   label=f'Competitive: n={int(chosen.sum())}',zorder=2)
        group_rows=[]
        if categorical:
            bins=[(i,label,x.eq(i)) for i,label in enumerate(reference['categories'])]
        else:
            edges=np.asarray(reference['edges'])
            bins=[(i,f'[{left:g}, {right:g}{"]" if i==len(edges)-2 else ")"}',
                   x.ge(left)&(x.le(right) if i==len(edges)-2 else x.lt(right)))
                  for i,(left,right) in enumerate(zip(edges,edges[1:]))]
        for i,label,mask in bins:
            sample=y[mask]
            position=float(i) if categorical else (reference['edges'][i]+reference['edges'][i+1])/2
            row=dict(parameter=name,panel=panel_index,panel_kind=panel['kind'],objective=panel['label'],
                     bin_or_category=label,position=position,n=len(sample),supported=len(sample)>=min_bin_models,
                     low=sample.quantile(lower_quantile) if len(sample) else np.nan,
                     median=sample.median(),high=sample.quantile(1-lower_quantile) if len(sample) else np.nan)
            group_rows.append(row);summaries.append(row)
        summary=pd.DataFrame(group_rows)
        good=summary.supported
        # Gaps remain gaps: sparse/empty bins are not bridged by fitted curves.
        for column,color,label in [('median','#d95f02','Bin median'),('low','#238b45',f'Bin {100*lower_quantile:g}th percentile')]:
            ordinate=summary[column].where(good)
            ax.plot(summary.position,ordinate,marker='s' if categorical else None,
                    linestyle='none' if categorical else '-',color=color,markersize=5,label=label,zorder=3)
        if categorical:
            ax.vlines(summary.position[good],summary.low[good],summary.high[good],color='#d95f02',lw=1.5)
            labels=[f'{label}\nn={int(n)}' for label,n in zip(reference['categories'],summary.n)]
            ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right')
        else:
            for row in summary.itertuples():
                ax.text(row.position,.01,str(row.n),transform=ax.get_xaxis_transform(),ha='center',va='bottom',
                        fontsize=6,color='#555555')
        # Keep true zero/negative scores visible by falling back to linear scale.
        effective_scale=score_scale if score_scale=='linear' or y.gt(0).all() else 'linear'
        ax.set_yscale(effective_scale)
        units='log10' if reference['kind']=='log10' else 'saved units'
        ax.set_xlabel(parameter_label(name) if categorical else f'{parameter_label(name)} ({units})')
        ax.set_ylabel('Added score = expanded − previous' if panel['kind']=='added' else 'Saved score (lower is better)')
        title=f"{panel['kind'].capitalize()}: {panel['label']}"
        if panel['kind']=='added':title+='\nBlue: competitive for '+panel['selection_objective']
        if score_scale!=effective_scale:title+='\nLinear scale: nonpositive score present'
        ax.set_title(fill(title,58,replace_whitespace=False),fontsize=9)
        ax.legend(fontsize=6)
        audits.append(dict(parameter=name,panel=panel_index,kind=panel['kind'],objective=panel['label'],
                           selection_objective=panel['selection_objective'],selection_cutoff=panel['cutoff'],
                           n_pool=len(candidates),n_valid_parameter=len(x),n_competitive=int(panel['mask'].sum()),
                           n_competitive_valid_parameter=int(chosen.sum()),score_scale=effective_scale))
    ax=axes.flat[len(panels)]
    if categorical:
        categories=reference['categories'];width=.8/len(sequence)
        for i,objective in enumerate(sequence):
            sample=values[selected[objective].reindex(values.index)]
            frequencies=sample.value_counts(normalize=True).reindex(categories,fill_value=0)
            ax.bar(np.arange(len(categories))+(i-(len(sequence)-1)/2)*width,frequencies,width=width,
                   alpha=.8,label=f'{objective}: n={len(sample)}')
        ax.set_xticks(range(len(categories)),categories,rotation=35,ha='right')
        ax.set(xlabel=parameter_label(name),ylabel='Selected fraction per category',title='Selected distributions: categories')
    else:
        for i,objective in enumerate(['Common pool',*sequence]):
            sample=x if i==0 else x[selected[objective].reindex(x.index)]
            ordered=np.sort(sample.to_numpy())
            if len(ordered):
                # Consolidate ties so each jump has the exact empirical mass.
                locations,counts=np.unique(ordered,return_counts=True)
                locations=np.r_[locations[0],locations]
                cumulative=np.r_[0,np.cumsum(counts)/len(ordered)]
                ax.step(locations,cumulative,where='post',color='black' if i==0 else f'C{(i-1)%10}',
                        linestyle='--' if i==0 else '-',label=f'{objective}: n={len(ordered)}')
        ax.set(xlabel=f'{parameter_label(name)} ({"log10" if reference["kind"]=="log10" else "saved units"})',
               ylabel='Empirical cumulative fraction',ylim=(0,1.02),title='Selected distributions: ECDF')
    ax.legend(fontsize=6)
    for ax in axes.flat[len(panels)+1:]:ax.set_visible(False)
    fig.suptitle(f'{parameter_label(name)}: observable preferences and parameter restriction\n'
                 f'Relative allowance={allowance:g}; absolute allowance={absolute_allowance:g}; '
                 f'bin summaries require n≥{min_bin_models}',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95))
    return fig,pd.DataFrame(audits),pd.DataFrame(summaries)
