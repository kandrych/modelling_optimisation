"""Read-only diagnostics for saved optimiser evaluations, not likelihood inference.

All models retain their unique evaluation IDs. No simulation or observation
processing code is imported. Units and original score contributions are retained.
"""
from itertools import combinations
import json
import math

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import wasserstein_distance

from observation_convergence_utils import parameter_values, parameter_metrics
from run_summary_utils import INTERFEROMETRY, PDI, digest
from parameter_labels import parameter_label

GROUPS = {'SED': ['score_sed'], 'INT': INTERFEROMETRY,
          'PDI': PDI, 'ALMA': ['score_alma']}


def objective_design():
    """SED-anchored full factorial plus single groups and full-minus-SED.

    Labels denote rescoring experiments, not historical execution sequences.
    INT mode and PDI bands are taken from the selected compatible pool.
    """
    result = {'SED': ['SED']}
    for size in range(1, 4):
        for extra in combinations(['INT', 'PDI', 'ALMA'], size):
            result['SED+' + '+'.join(extra)] = ['SED', *extra]
    result['All'] = result.pop('SED+INT+PDI+ALMA')
    result.update({name: [name] for name in ['INT', 'PDI', 'ALMA']})
    result['INT+PDI+ALMA'] = ['INT', 'PDI', 'ALMA']
    return result


def score_matrix(pool, objectives):
    """Sum each unique included term once; reject missing/nonfinite scores."""
    scores = pd.DataFrame(index=pool.index)
    for label, groups in objectives.items():
        columns = list(dict.fromkeys(c for group in groups for c in GROUPS[group]))
        if not np.isfinite(pool[columns].to_numpy(float)).all():
            raise ValueError(f'{label}: requested score vector is incomplete/nonfinite.')
        scores[label] = pool[columns].sum(axis=1, min_count=len(columns))
    return scores


def competitive_mask(scores, mode, level, absolute_allowance=0.):
    """Include ties. Allowance is an objective tolerance, not a probability."""
    if scores.empty or not np.isfinite(scores).all():
        raise ValueError('Selection needs nonempty finite scores.')
    if not np.isfinite(level) or not np.isfinite(absolute_allowance) or absolute_allowance < 0:
        raise ValueError('Invalid selection level or absolute allowance.')
    if mode == 'fraction':
        if not 0 < level <= 1:
            raise ValueError('Fraction must be in (0, 1].')
        cutoff = scores.nsmallest(math.ceil(len(scores)*level)).max()
    elif mode == 'allowance':
        if level < 0:
            raise ValueError('Relative allowance must be nonnegative.')
        cutoff = scores.min() + max(abs(scores.min())*level, absolute_allowance)
    else:
        raise ValueError('Use allowance or fraction.')
    return scores.le(cutoff), float(cutoff)


def addition_design(objectives):
    """Every one-group addition in the anchored factorial and each leave-one-out."""
    rows = []
    for before, old in objectives.items():
        for after, new in objectives.items():
            added = set(new) - set(old)
            if set(old) < set(new) and len(added) == 1:
                rows.append(dict(before=before, after=after, added=next(iter(added))))
    return pd.DataFrame(rows)


def distribution_effect(before, after, name, reference):
    """Distribution shift and restriction are deliberately separate quantities.

    Numeric Wasserstein distance uses fixed allowed-span normalisation (log
    coordinates for log-sampled parameters). Category TV is half the L1 distance.
    Neither distance is a significance test or a posterior information gain.
    """
    old = parameter_values(before, name, reference)
    new = parameter_values(after, name, reference)
    result = dict(n_before=len(old), n_after=len(new),
                  shift=np.nan, width_ratio=np.nan, median_shift=np.nan)
    if old.empty or new.empty:
        return result
    if reference['kind'] == 'categorical':
        categories = reference['categories']
        p = old.value_counts(normalize=True).reindex(categories, fill_value=0)
        q = new.value_counts(normalize=True).reindex(categories, fill_value=0)
        result['shift'] = float(abs(p-q).sum()/2)
    else:
        span = reference['upper'] - reference['lower']
        if not np.isfinite(span) or span <= 0:
            raise ValueError(f'{name}: finite positive bounds required for distance.')
        result['shift'] = wasserstein_distance(old, new)/span
        old_width = old.quantile(.9)-old.quantile(.1)
        new_width = new.quantile(.9)-new.quantile(.1)
        result['width_ratio'] = new_width/old_width if old_width > 0 else np.nan
        result['median_shift'] = (new.median()-old.median())/span
    return result


def restriction_tables(pool, scores, definitions, references, additions,
                       selection_levels, absolute_allowance=0., min_models=20):
    """Same-pool analysis with pooled and within-history controls.

    Each history is rescored on its own fixed candidates; it is not a replicate
    unless the experiment, numerical fidelity and optimiser protocol also match.
    """
    metrics, counts, effects = [], [], []
    scopes = [('pooled', pool), *pool.groupby('history', sort=False)]
    for scope, candidates in scopes:
        for mode, levels in selection_levels.items():
            for level in levels:
                selected = {}
                for objective in scores:
                    values = scores.loc[candidates.index, objective]
                    mask, cutoff = competitive_mask(values, mode, level, absolute_allowance)
                    sample = candidates.loc[mask]
                    selected[objective] = sample
                    meta = dict(scope=scope, mode=mode, level=level, objective=objective)
                    counts.append(dict(meta, n_pool=len(candidates), n_selected=len(sample),
                                       cutoff=cutoff, best_score=values.min()))
                    for name, reference in references.items():
                        row, _, _ = parameter_metrics(candidates, sample, name,
                            definitions[name], min_models=min_models, reference=reference)
                        row['best_parameter_values'] = json.dumps(candidates.loc[
                            values.eq(values.min()), 'param__'+name].dropna().unique().tolist())
                        metrics.append(dict(meta, **row))
                for addition in additions.itertuples(index=False):
                    old, new = selected[addition.before], selected[addition.after]
                    old_ids, new_ids = set(old.evaluation_id), set(new.evaluation_id)
                    jaccard = len(old_ids & new_ids)/len(old_ids | new_ids)
                    for name, reference in references.items():
                        effect = distribution_effect(old, new, name, reference)
                        effects.append(dict(scope=scope, mode=mode, level=level,
                            before=addition.before, after=addition.after, added=addition.added,
                            parameter=name, kind=reference['kind'], sample_jaccard=jaccard,
                            supported=min(effect['n_before'], effect['n_after']) >= min_models,
                            **effect))
    return pd.DataFrame(metrics), pd.DataFrame(counts), pd.DataFrame(effects)


def association_table(pool, selected, references, min_models=20):
    """Paired numeric Spearman rho; no p-values for adaptive optimiser samples."""
    names = [name for name, ref in references.items() if ref['kind'] != 'categorical']
    rows = []
    for x, y in combinations(names, 2):
        row = dict(x=x, y=y)
        for label, frame in [('all', pool), ('selected', selected)]:
            paired = pd.concat([parameter_values(frame, x, references[x]).rename('x'),
                                parameter_values(frame, y, references[y]).rename('y')], axis=1).dropna()
            supported = len(paired) >= min_models and paired.x.nunique() > 1 and paired.y.nunique() > 1
            row['n_'+label] = len(paired)
            row['rho_'+label] = paired.x.corr(paired.y, method='spearman') if supported else np.nan
        row['change_in_rho'] = row['rho_selected']-row['rho_all']
        rows.append(row)
    return pd.DataFrame(rows)


def mixed_distance(pool, references):
    """Mean per-parameter distance: span-scaled numeric, 0/1 categorical.

    Equal parameter weights are a diagnostic convention, not physical equivalence.
    Missing/invalid values and out-of-bound numeric values are rejected explicitly.
    PA is ordinary linear numeric, as requested. No duplicate models are merged.
    """
    if not references:
        raise ValueError('At least one varied parameter required.')
    distance = np.zeros((len(pool), len(pool)))
    for name, ref in references.items():
        values = parameter_values(pool, name, ref).reindex(pool.index)
        if values.isna().any():
            raise ValueError(f'{name}: missing/invalid distance coordinate.')
        if ref['kind'] == 'categorical':
            a = values.to_numpy()
            distance += (a[:, None] != a[None, :]).astype(float)
        else:
            span = ref['upper']-ref['lower']
            if not np.isfinite(span) or span <= 0:
                raise ValueError('Distance requires finite nonconstant parameter bounds.')
            a = (values.to_numpy(float)-ref['lower'])/span
            if np.any((a < -1e-10) | (a > 1+1e-10)):
                raise ValueError(f'{name}: value outside saved bounds.')
            distance += abs(a[:, None]-a[None, :])
    return distance/len(references)


def alternative_families(pool, scores, references, allowance=.1,
                         absolute_allowance=0., distance_cuts=(.10, .20, .30)):
    """Complete-linkage families at multiple arbitrary distance resolutions.

    Counts describe sampled families. One connected family does NOT prove uniqueness;
    absence of sampled alternatives does NOT exclude undiscovered alternatives.
    """
    keep, cutoff = competitive_mask(scores, 'allowance', allowance, absolute_allowance)
    candidates = pool.loc[keep].copy()
    candidates['objective_score'] = scores.loc[keep]
    distance = mixed_distance(candidates, references)
    best_position = int(np.argmin(candidates.objective_score.to_numpy()))
    candidates['distance_to_best'] = distance[:, best_position]
    candidates['score_excess'] = candidates.objective_score-scores.min()
    tree = linkage(squareform(distance, checks=False), method='complete') if len(candidates)>1 else None
    members, summaries = [], []
    for cut in distance_cuts:
        if cut <= 0:
            raise ValueError('Family distance cut must be positive.')
        labels = fcluster(tree, cut, criterion='distance') if tree is not None else np.ones(1, int)
        frame = candidates.assign(distance_cut=cut, family=labels)
        members.append(frame)
        for family, rows in frame.groupby('family'):
            best_score = rows.objective_score.min()
            summaries.append(dict(distance_cut=cut, family=family, n=len(rows),
                best_score=best_score, best_ids=json.dumps(rows.loc[
                    rows.objective_score.eq(best_score), 'evaluation_id'].tolist()),
                histories=' | '.join(sorted(rows.history.unique())),
                largest_distance_to_best=rows.distance_to_best.max(), cutoff=cutoff))
    return pd.concat(members, ignore_index=True), pd.DataFrame(summaries)


def pareto_mask(group_scores):
    """Exact nondominated candidates: lower is better in every score group."""
    array = np.asarray(group_scores, float)
    if not np.isfinite(array).all():
        raise ValueError('Pareto analysis requires complete finite score vectors.')
    return np.array([not np.any(np.all(array <= row, axis=1) &
                               np.any(array < row, axis=1)) for row in array])


def empirical_profiles(pool, scores, references, bins_minimum=10):
    """Sampled bin minima and lower deciles; NOT nuisance-reoptimised profiles."""
    rows = []
    for name, ref in references.items():
        values = parameter_values(pool, name, ref)
        if ref['kind'] == 'categorical':
            groups = [(str(value), i, values.eq(value)) for i, value in enumerate(ref['categories'])]
        else:
            edges = ref['edges']
            groups = [(f'[{a:g}, {b:g}{"]" if i==len(edges)-2 else ")"}', (a+b)/2,
                       values.ge(a) & (values.le(b) if i==len(edges)-2 else values.lt(b)))
                      for i, (a,b) in enumerate(zip(edges, edges[1:]))]
        for label, position, mask in groups:
            for objective in scores:
                sample = scores.loc[values.index[mask], objective]
                rows.append(dict(parameter=name, kind=ref['kind'], bin=label, position=position,
                    objective=objective, n=len(sample), supported=len(sample)>=bins_minimum,
                    minimum=sample.min(), q10=sample.quantile(.1),
                    excess_minimum=sample.min()-scores[objective].min()))
    return pd.DataFrame(rows)


def bootstrap_history_effects(effects, histories, repetitions=1000, seed=42):
    """Bootstrap the mean effect over verified comparable independent runs.

    Caller must explicitly certify history membership. Uses history-level effects,
    not trial resampling. Intervals concern between-run repeatability conditional
    on this design, NOT physical parameter confidence intervals.
    """
    if len(set(histories)) < 5:
        raise ValueError('At least five verified independent comparable histories required.')
    data = effects[effects.scope.isin(histories) & effects.supported]
    rng = np.random.default_rng(seed)
    rows = []
    keys = ['before', 'after', 'added', 'parameter', 'mode', 'level']
    for labels, group in data.groupby(keys):
        if group.scope.nunique() != len(set(histories)):
            continue
        values = group.set_index('scope').loc[sorted(set(histories)), 'shift'].to_numpy()
        means = rng.choice(values, (repetitions, len(values)), replace=True).mean(axis=1)
        rows.append(dict(zip(keys, labels), n_histories=len(values), mean_shift=values.mean(),
                         bootstrap_q025=np.quantile(means,.025), bootstrap_q975=np.quantile(means,.975)))
    return pd.DataFrame(rows)


def plot_heatmap(table, title, label, cmap='viridis'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(8, .9*len(table.columns)), max(4, .35*len(table))))
    image = ax.imshow(table.to_numpy(float), aspect='auto', cmap=cmap)
    ax.set_xticks(range(len(table.columns)), [parameter_label(name) for name in table.columns], rotation=40, ha='right')
    ax.set_yticks(range(len(table)), [parameter_label(name) for name in table.index])
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=label)
    fig.tight_layout()
    return fig


def eligible_pool_candidates(evaluations, inventory, records):
    """Final-budget, native-cost-audited candidates grouped by physical setup.

    Observable availability is not a physical distinction. Observable-specific
    scoring compatibility is checked separately by available_score_pools.
    """
    data = evaluations.loc[evaluations.usable & evaluations.cost_terms_match
        & evaluations.known_physical_metadata
        & np.isfinite(evaluations.cost)
        & np.isclose(evaluations.budget, evaluations.configured_max_budget)].copy()
    if not data.evaluation_id.is_unique:
        raise ValueError('Evaluation IDs must be unique; repeated configurations remain separate.')
    histories = inventory.set_index('history')
    def key(row):
        meta = histories.loc[row.history]
        scoring = records[row.evaluation_id]['scoring_metadata']
        return digest(dict(space=meta.configspace_hash, template=meta.template_hash,
            photons=meta.photon_settings, numerical=scoring['numerical_fidelity']))
    data['physical_pool_key'] = [key(row) for row in data.itertuples()]
    labels = {key: f'Physical pool {i:02d}'
              for i, key in enumerate(sorted(data.physical_pool_key.unique()), 1)}
    data['physical_pool'] = data.physical_pool_key.map(labels)
    return data


def available_score_pools(candidates, records, groups):
    """Partition finite-score candidates by metadata relevant to these groups.

    The caller supplies one physical/numerical pool. Unneeded observable
    descriptors are ignored, but relevant unknown descriptors stay explicit.
    This is recorded compatibility evidence, not proof of historical identity.
    """
    if candidates.physical_pool_key.nunique() > 1:
        raise ValueError('Select one physical/numerical pool first.')
    columns = list(dict.fromkeys(c for group in groups for c in GROUPS[group]))
    valid = np.isfinite(candidates.reindex(columns=columns).to_numpy(float)).all(axis=1)
    data = candidates.loc[valid].copy()
    def key(row):
        meta = records[row.evaluation_id]['scoring_metadata']
        relevant = {'physical_pool_key': row.physical_pool_key}
        if 'INT' in groups:
            relevant['interferometry'] = meta['interferometry']
            relevant['interferometry_products'] = sorted(
                product for product in meta['products'] if product.startswith('vis2_'))
        if 'PDI' in groups:
            relevant['pdi'] = meta['pdi']
        if 'ALMA' in groups:
            relevant['alma_observed'] = meta['alma_observed']
        # The existing loader records no additional SED-specific data fingerprint.
        return digest(relevant)
    data['score_pool_key'] = [key(row) for row in data.itertuples()]
    return {key: rows for key, rows in data.groupby('score_pool_key', sort=True)}


def analyse_pool_views(candidates, records, objectives, definitions, references,
                       selection_levels, absolute_allowance=0., min_models=20):
    """Available-per-objective summaries and same-candidate addition effects.

    No cross-objective effect is computed between different available pools.
    Every effect uses the intersection required by its comparison. Keep all
    ties/repeated evaluations and audit empty requests without inventing scores.
    """
    tables = {name: [] for name in ('availability', 'members', 'winners',
                                    'metrics', 'counts', 'effects')}
    additions = addition_design(objectives)
    requests = [('available_per_objective', label, {label: groups}, additions.iloc[:0])
                for label, groups in objectives.items()]
    for row in additions.itertuples(index=False):
        requests.append(('shared_per_comparison', row.before+' → '+row.after,
            {name: objectives[name] for name in (row.before, row.after)},
            pd.DataFrame([row._asdict()])))
    for view, comparison, design, contrasts in requests:
        groups = list(dict.fromkeys(g for values in design.values() for g in values))
        pools = available_score_pools(candidates, records, groups)
        tables['availability'].append(pd.DataFrame([dict(view=view, comparison=comparison,
            n_eligible=sum(len(frame) for frame in pools.values()), n_score_pools=len(pools))]))
        for key, frame in pools.items():
            meta = dict(view=view, comparison=comparison, score_pool_key=key)
            scores = score_matrix(frame, design)
            tables['members'].append(frame.assign(**meta))
            for objective in scores:
                best = scores[objective].min()
                winners = frame.loc[scores[objective].eq(best)].copy()
                winners['selected_by'] = objective
                winners['objective_score'] = best
                tables['winners'].append(winners.assign(**meta))
            metrics, counts, effects = restriction_tables(frame, scores, definitions,
                references, contrasts, selection_levels, absolute_allowance, min_models)
            for name, table in [('metrics', metrics), ('counts', counts), ('effects', effects)]:
                if not table.empty:
                    tables[name].append(table.assign(**meta))
    return {name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            for name, parts in tables.items()}


def plot_available_pool_costs(pool, objective, groups, references,
                              allowances=(.1, .5, 1.), panels_per_page=6,
                              point_size=16, point_alpha=.55):
    """Cost versus every parameter, using one objective-specific score pool.

    Allowances are relative to abs(best); they are not probabilities. Nested
    selections are drawn broadest first. Each evaluation has equal visual weight.
    Returns paginated figures and explicit cutoff/coverage tables.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if panels_per_page < 1 or len(allowances) != 3 or not all(
            np.isfinite(v) and v >= 0 for v in allowances):
        raise ValueError('Use three finite nonnegative allowances and a positive page size.')
    if pool.empty:
        return [], pd.DataFrame(), pd.DataFrame()
    if pool.score_pool_key.nunique() != 1 or not pool.evaluation_id.is_unique:
        raise ValueError('Supply one score pool with unique evaluation IDs.')
    scores = score_matrix(pool, {objective: groups})[objective]
    best = float(scores.min())
    levels = sorted(set(allowances), reverse=True)
    if len(levels) != 3:
        raise ValueError('The three allowances must be distinct.')
    colors = ['#9ecae1', '#2171b5', '#d95f02']
    masks, cuts, summary = {}, {}, []
    for level in levels:
        mask, cutoff = competitive_mask(scores, 'allowance', level, 0.)
        masks[level], cuts[level] = mask, cutoff
        summary.append(dict(objective=objective, score_pool_key=pool.score_pool_key.iloc[0],
            allowance=level, best_score=best, cutoff=cutoff, n_pool=len(pool),
            n_selected=int(mask.sum()), n_unique_configurations=pool.parameter_hash.nunique(),
            n_histories=pool.history.nunique()))
    names = list(references)
    pages, coverage = [], []
    scale = 'log' if scores.gt(0).all() else 'linear'
    for start in range(0, len(names), panels_per_page):
        page_names = names[start:start+panels_per_page]
        ncols = min(3, len(page_names))
        nrows = math.ceil(len(page_names)/ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.7*nrows+1.5),
                                 squeeze=False, sharey=True)
        for ax, name in zip(axes.flat, page_names):
            ref = references[name]
            values = parameter_values(pool, name, ref)
            if ref['kind'] == 'categorical':
                positions = {str(value): i for i, value in enumerate(ref['categories'])}
                x = values.map(positions)
                ax.set_xticks(range(len(positions)), list(positions), rotation=35, ha='right')
                coordinate = 'categories'
            else:
                x = values
                coordinate = 'log10' if ref['kind'] == 'log10' else 'saved units'
            valid = np.isfinite(x)
            x = x.loc[valid]
            y = scores.loc[x.index]
            ax.scatter(x, y, s=point_size, alpha=point_alpha, color='#777777', linewidths=0)
            for level, color in zip(levels, colors):
                selected = masks[level].loc[x.index]
                ax.scatter(x.loc[selected], y.loc[selected], s=point_size,
                           alpha=point_alpha, color=color, linewidths=0)
                ax.axhline(cuts[level], color=color, linestyle='--', linewidth=.8, alpha=.8)
            ax.set_yscale(scale)
            ax.set(xlabel=f'{parameter_label(name)} ({coordinate})',
                   ylabel=f'{objective} cost' + (' (log axis)' if scale == 'log' else ''),
                   title=f'{parameter_label(name)} — valid n={len(x)} / {len(pool)}')
            ax.grid(alpha=.15)
            coverage.append(dict(objective=objective,score_pool_key=pool.score_pool_key.iloc[0],
                parameter=name,n_pool=len(pool),n_valid=len(x),n_missing_or_invalid=len(pool)-len(x),
                **{f'n_within_{level:g}':int(masks[level].loc[x.index].sum()) for level in levels}))
        for ax in axes.flat[len(page_names):]:ax.set_visible(False)
        handles = [Line2D([], [], marker='o', linestyle='', color='#777777', label=f'All evaluations: n={len(pool)}')]
        handles += [Line2D([], [], marker='o', linestyle='--', color=color,
            label=f'Within {level:.0%}: cost ≤ {cuts[level]:.4g}; n={int(masks[level].sum())}')
            for level,color in zip(levels,colors)]
        fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9)
        fig.suptitle(f'View A — {objective} | score pool {pool.score_pool_key.iloc[0][:12]}\n'
            f'Best cost={best:.5g}; {pool.parameter_hash.nunique()} configurations; '
            f'{pool.history.nunique()} histories | page {start//panels_per_page+1}/{math.ceil(len(names)/panels_per_page)}\n'
            'Nested selections: tighter colors cover broader ones; dashed lines mark cutoffs', fontsize=11)
        fig.tight_layout(rect=(0,.10,1,.88))
        pages.append(fig)
    return pages, pd.DataFrame(summary), pd.DataFrame(coverage)
