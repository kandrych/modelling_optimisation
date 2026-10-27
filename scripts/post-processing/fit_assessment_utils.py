"""Read-only fit assessment of a saved SMAC snapshot.

Reuse the audited pools in model_analysis_utils. Scores, uncertainties and model
files are never changed. Selection fractions, exponential weights and families
are descriptive conventions for adaptively sampled models, not probabilities.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from model_analysis_utils import (
    GROUPS, available_score_pools, competitive_mask, distribution_effect,
    alternative_families, score_matrix,
)
from observation_convergence_utils import parameter_reference, parameter_values
from parameter_labels import parameter_description, parameter_label
from run_summary_utils import resolve_model_files, prediction_tables, file_hash


def parameter_setup(pool, spaces):
    """Use the selected physical pool's saved definitions, including categories."""
    definitions = {p['name']: p for p in spaces[pool.iloc[0].history]['hyperparameters']}
    definitions = {name: definition for name, definition in definitions.items()
                   if 'param__' + name in pool and definition.get('type') != 'constant'}
    references = {name: parameter_reference(pool, name, definition)
                  for name, definition in definitions.items()}
    glossary = pd.DataFrame([
        dict(parameter=name, meaning=parameter_description(name),
             coordinates=references[name]['kind'],
             bounds_or_choices=str(definition.get('choices', definition.get('sequence',
                 [definition.get('lower'), definition.get('upper')]))),
             sampled_values=pool['param__' + name].nunique(),
             inactive_or_missing=int(pool['param__' + name].isna().sum()))
        for name, definition in definitions.items()])
    return definitions, references, glossary


def objective_pools(candidates, records, objectives):
    """Keep all scoring partitions visible; never rank across incompatible pools."""
    pools, rows = {}, []
    for objective, groups in objectives.items():
        pools[objective] = available_score_pools(candidates, records, groups)
        for key, pool in pools[objective].items():
            rows.append(dict(objective=objective, score_pool_key=key,
                evaluations=len(pool), configurations=pool.parameter_hash.nunique(),
                histories=pool.history.nunique()))
        if not pools[objective]:
            rows.append(dict(objective=objective, score_pool_key='unavailable',
                             evaluations=0, configurations=0, histories=0))
    return pools, pd.DataFrame(rows)


def choose_pool(pools, key=None):
    """Default to the largest partition; report the choice in notebook output."""
    if not pools:
        return pd.DataFrame()
    if key is None:
        key = min(pools, key=lambda item: (-len(pools[item]), item))
    if key not in pools:
        raise ValueError('Requested scoring-pool key is not available.')
    return pools[key].copy()


def winner(pool, groups):
    scores = score_matrix(pool, {'objective': groups}).objective
    ties = pool.loc[scores.eq(scores.min())].sort_values('evaluation_id')
    return ties.iloc[0].evaluation_id, len(ties), float(scores.min())


def representatives(pools, focus, objectives, references, family_allowance=.5,
                    family_distance=.2, max_alternatives=3, max_family_models=2000):
    """One deterministic exemplar per tied minimum; preserve tie counts and IDs."""
    roles, family_members = [], pd.DataFrame()
    family_note = 'No joint pool available.'
    def add(pool, groups, role):
        model, ties, score = winner(pool, groups)
        values = score_matrix(pool, {'score': groups}).score
        roles.append(dict(evaluation_id=model, role=role, score=score,
            tied_minima=ties, tied_ids=json.dumps(pool.loc[values.eq(score), 'evaluation_id'].tolist()),
            score_pool_key=pool.iloc[0].score_pool_key, n_pool=len(pool)))
    if not focus.empty:
        add(focus, objectives['All'], 'Current joint minimum')
        scores = score_matrix(focus, {'All': objectives['All']}).All
        selected, _ = competitive_mask(scores, 'allowance', family_allowance)
        if selected.sum() > max_family_models:
            family_note = f'Family clustering skipped: {selected.sum()} models exceeds MAX_FAMILY_MODELS.'
        elif any(parameter_values(focus.loc[selected], name, ref).size != selected.sum()
                 for name, ref in references.items()):
            family_note = 'Family clustering unavailable: inactive or invalid parameter coordinates.'
        else:
            # Fixed parameters add no separation and should not dilute distances.
            varied = {name: ref for name, ref in references.items()
                      if focus.loc[selected, 'param__' + name].nunique() > 1}
            if varied:
                family_members, families = alternative_families(
                    focus, scores, varied, allowance=family_allowance,
                    distance_cuts=(family_distance,))
                best_id = roles[0]['evaluation_id']
                best_family = family_members.loc[family_members.evaluation_id.eq(best_id), 'family'].iloc[0]
                alternatives = families.loc[families.family.ne(best_family)].sort_values('best_score')
                for family in alternatives.head(max_alternatives).family:
                    add(family_members.loc[family_members.family.eq(family)], objectives['All'],
                        f'Sampled alternative family {family}')
                family_note = (f'{len(families)} sampled families among {selected.sum()} models within '
                               f'{family_allowance:.0%} of the joint minimum; '
                               f'{min(len(alternatives), max_alternatives)} alternative representatives shown. '
                               'Distance uses only parameters varying in this selected family sample.')
            else:
                family_note = 'One sampled configuration; no separated families can be assessed.'
    for objective in ('SED', 'INT', 'PDI', 'ALMA'):
        for key, pool in pools.get(objective, {}).items():
            add(pool, objectives[objective], f'{objective} minimum / pool {key[:8]}')
    return pd.DataFrame(roles), family_members, family_note


def resolve_diagnostics(selection, records):
    """Match configuration/fidelity AND stochastic realization where recorded.

    A unique legacy configuration match can be displayed, but is explicitly
    unverified. Ambiguous folders or mismatched seeds/costs are never guessed.
    """
    rows, folders = [], {}
    for _, row in selection.iterrows():
        matches = resolve_model_files(row, records)
        exact, legacy, rejected = [], [], []
        for match in matches:
            folder = match['folder']
            status_path = folder / 'trial_status.json'
            if not status_path.exists():
                legacy.append(folder)
                continue
            try:
                status = json.loads(status_path.read_text())
                instance = json.loads(row.instance) if isinstance(row.instance, str) else row.instance
                same = (status.get('seed') == row.seed and status.get('instance') == instance
                        and np.isclose(float(status.get('cost', np.nan)), row.cost,
                                       rtol=1e-8, atol=1e-8))
                (exact if same else rejected).append(folder)
            except (ValueError, TypeError, OSError) as error:
                rejected.append(f'{folder}: {error}')
        if len(exact) == 1:
            folder, status = exact[0], 'verified realization'
        elif not exact and len(legacy) == 1 and len(matches) == 1:
            folder, status = legacy[0], 'configuration/fidelity only; realization unverified'
        else:
            folder = None
            status = 'ambiguous folders' if len(exact) > 1 or len(legacy) > 1 else 'no matching saved realization'
        if folder is not None:
            folders[row.evaluation_id] = folder
        rows.append(dict(evaluation_id=row.evaluation_id, status=status,
                         folder=str(folder) if folder else None,
                         configuration_matches=len(matches), rejected_matches=len(rejected)))
    return folders, pd.DataFrame(rows)


def sed_residuals(folder, sed_info, photometry, reddening_law, secondary=True,
                  companion_temperature=4000., companion_fraction=.039,
                  companion_reference_um=1.65, rtol=1e-5):
    """Reproduce the current obriy_sed recipe at SAVED E(B-V), without refitting.

    Flux units: lambda F_lambda [erg/s/cm²]. Point residuals: (data-model)/sigma.
    Current observation files are not guaranteed historical originals: a score
    mismatch returns no usable residuals, with the discrepancy in the audit.
    """
    from astropy.io import fits
    columns = ['meas', 'e_meas', 'flag', 'unit', 'photband', 'source', '_r', '_RAJ2000',
               '_DEJ2000', 'cwave', 'cmeas', 'e_cmeas', 'cunit', 'color', 'include',
               'phase', 'bibcode', 'comments']
    data = pd.read_csv(photometry, sep=r'\s+', header=3, names=columns)
    data = data.loc[data.cwave.notna()].copy()  # Same selection as load_sed_data.
    x = data.cwave.to_numpy(float) * 1e-4  # Angstrom -> micrometres.
    observed = data.cwave.to_numpy(float) * data.cmeas.to_numpy(float)
    error = data.cwave.to_numpy(float) * data.e_cmeas.to_numpy(float)
    with fits.open(Path(folder) / 'data_th/sed_rt.fits.gz') as hdus:
        wave = np.asarray(hdus[1].data, float).copy()
        model = np.asarray(hdus[0].data[0, 0, 0, :], float).copy() * 1e3
    if secondary:
        if companion_temperature <= 0 or not 0 <= companion_fraction < 1:
            raise ValueError('Invalid companion temperature or flux fraction.')
        def blackbody(wavelength):
            metres = np.asarray(wavelength) * 1e-6
            return (2 * 6.62607015e-34 * 2.99792458e8**2 / metres**4
                    / np.expm1(6.62607015e-34 * 2.99792458e8 /
                               (metres * 1.380649e-23 * companion_temperature)))
        amplitude = companion_fraction / (1 - companion_fraction) * np.interp(
            companion_reference_um, wave, model)
        model += amplitude * blackbody(wave) / blackbody(companion_reference_um)
    law = pd.read_csv(reddening_law, sep=r'\s+', header=2, names=['WAVE', 'A/E'])
    extinction = interp1d(law.WAVE.to_numpy(float) * 1e-4, law['A/E'],
                         bounds_error=False, fill_value=0)(wave)
    model *= 10**(-extinction * float(sed_info['ebminv']) / 2.5)
    prediction = interp1d(wave, model, bounds_error=True)(x)
    if not np.isfinite(np.array([observed, error, prediction])).all() or (error <= 0).any():
        raise ValueError('SED points contain nonfinite values or nonpositive uncertainties.')
    residual = (observed - prediction) / error
    chi2 = float(residual @ residual)
    saved = float(sed_info['chi2'])
    matches = bool(np.isclose(chi2, saved, rtol=rtol, atol=1e-8))
    audit = dict(status='score verified' if matches else 'SED recipe/data mismatch',
                 chi2_reconstructed=chi2, chi2_saved=saved, n_points=len(x),
                 residual_rms=float(np.sqrt(chi2 / len(x))), ebminv=sed_info['ebminv'],
                 secondary=secondary, photometry_sha256=file_hash(photometry),
                 reddening_sha256=file_hash(reddening_law))
    points = pd.DataFrame(dict(wavelength_um=x, observed=observed, uncertainty=error,
                              model=prediction, residual_sigma=residual, band=data.photband.to_numpy()))
    return points if matches else pd.DataFrame(), audit


def load_sed_diagnostics(selection, records, folders, photometry, reddening_law,
                         recipe=None, recipe_by_history=None):
    tables, audits = {}, []
    for _, row in selection.iterrows():
        options = dict(recipe or {})
        options.update((recipe_by_history or {}).get(row.history, {}))
        try:
            if row.evaluation_id not in folders:
                raise FileNotFoundError('No matched saved model folder.')
            info = records[row.evaluation_id]['info'].get('sed')
            if not info:
                raise ValueError('No recorded SED score/recipe.')
            points, audit = sed_residuals(folders[row.evaluation_id], info, photometry,
                                         reddening_law, **options)
            if not points.empty:
                tables[row.evaluation_id] = points
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            audit = dict(status=f'Unavailable: {type(error).__name__}: {error}')
        audits.append(dict(evaluation_id=row.evaluation_id, **audit))
    return tables, pd.DataFrame(audits)


def fit_scorecard(selection, roles, records):
    """Recorded terms and physical amplitudes; missing predictions stay missing."""
    result = selection[['evaluation_id', 'model_id', 'history',
                        *[c for cols in GROUPS.values() for c in cols]]].copy()
    result['roles'] = result.evaluation_id.map(roles.groupby('evaluation_id').role.agg(' | '.join))
    pdi, alma = prediction_tables(selection, records)
    for band in 'VIH':
        if not pdi.empty:
            data = pdi.loc[pdi.band.eq(band)].set_index('evaluation_id')
            for key in ['model_over_observed', 'radial_loss', 'quadrant_loss']:
                result[f'PDI {band} {key}'] = result.evaluation_id.map(data[key])
    if not alma.empty:
        data = alma.set_index('evaluation_id')
        for key in ['model_to_observed_flux', 'residual_energy_loss']:
            result['ALMA ' + key] = result.evaluation_id.map(data[key])
    return result, pdi, alma


def _show_saved(ax, folder, pattern, title):
    import matplotlib.pyplot as plt
    paths = sorted(folder.glob(pattern)) if folder else []
    if len(paths) == 1:
        ax.imshow(plt.imread(paths[0]))
    else:
        reason = 'Missing saved diagnostic' if not paths else 'Ambiguous saved diagnostics'
        ax.text(.5, .5, reason, ha='center', va='center', transform=ax.transAxes)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def plot_fit_page(row, folder, sed_points=None, role=''):
    """Compact page: SED residuals, all four INT residual panels, PDI and ALMA.

    Raster diagnostic panels retain their original axes and normalization. Their
    individual file paths are shown separately in the notebook for larger views.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 2, figsize=(16, 25), constrained_layout=True)
    if sed_points is not None and not sed_points.empty:
        ax = axes[0, 0]
        ax.scatter(sed_points.wavelength_um, sed_points.residual_sigma, s=18, color='#86376e')
        ax.axhline(0, color='black', lw=1)
        ax.set(xscale='log', xlabel='Wavelength [µm]', ylabel='(data − model) / σ',
               title='SED residuals; saved E(B−V), verified χ²')
        ax.grid(alpha=.2)
    else:
        axes[0, 0].text(.5, .5, 'Verified SED residuals unavailable\nSee SED audit and original saved SED plot.',
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_axis_off()
    instruments = [('PIONIER', '1.65'), ('GRAVITY', '2.2'), ('MATISSE L', '3.5'), ('MATISSE N', '10.0')]
    for ax, (instrument, wave) in zip([axes[0, 1], *axes[1], axes[2, 0]], instruments):
        _show_saved(ax, folder, f'figures/visibilities_{instrument} {wave}.png',
                    f'{instrument}: visibility and (model − data) / σ')
    _show_saved(axes[2, 1], folder, 'figures/alma_comparison.png',
                'ALMA: data, model, residual / background RMS, radial profile')
    _show_saved(axes[3, 0], folder, 'figures/pdi_constraints_v.png',
                'PDI V: amplitude, radial shape and quadrants')
    _show_saved(axes[3, 1], folder, 'figures/pdi_constraints_i.png',
                'PDI I: amplitude, radial shape and quadrants')
    _show_saved(axes[4, 0], folder, 'figures/pdi_constraints_h.png',
                'PDI H: amplitude, radial shape and quadrants')
    if sed_points is not None and not sed_points.empty:
        ordered = sed_points.sort_values('wavelength_um')
        axes[4, 1].errorbar(ordered.wavelength_um, ordered.observed,
                           yerr=ordered.uncertainty, fmt='.', color='#2369a1', label='Observation')
        axes[4, 1].scatter(ordered.wavelength_um, ordered.model, s=20, marker='x', color='#bb365d', label='Model at data wavelengths')
        axes[4, 1].set(xscale='log', yscale='log', xlabel='Wavelength [µm]',
                       ylabel='λ Fλ [erg/s/cm²]', title='SED flux and saved reddening')
        axes[4, 1].legend(fontsize=8)
    else:
        _show_saved(axes[4, 1], folder, 'SED_Akke_model1_MCFOST.png', 'Original saved SED plot')
    fig.suptitle(f'{row.model_id} — {role}\n{row.history}', fontsize=13)
    return fig


def diagnostic_paths(folder):
    if folder is None:
        return []
    patterns = ['*SED*.png', 'figures/visibilities_*.png',
                'figures/pdi_constraints_*.png', 'figures/alma_comparison.png']
    return sorted(set(path for pattern in patterns for path in folder.glob(pattern)))


def analysis_sample(pool, scores, fraction, counting='evaluations'):
    """Unique counting uses mean saved score per configuration BEFORE selection.

    This is a robustness experiment, never the score assigned to a saved model.
    Original evaluation IDs and scores in the input remain unchanged.
    """
    data = pool.copy()
    data['analysis_score'] = scores
    if counting == 'configurations':
        averages = data.groupby('parameter_hash').analysis_score.mean()
        data = data.sort_values('evaluation_id').drop_duplicates('parameter_hash').copy()
        data['analysis_score'] = data.parameter_hash.map(averages)
    elif counting != 'evaluations':
        raise ValueError('Counting must be evaluations or configurations.')
    mask, cutoff = competitive_mask(data.analysis_score, 'fraction', fraction)
    return data.loc[mask].copy(), cutoff


def sample_weights(sample, half_fraction=None, absolute_delta=None):
    """Weight halves at J−Jmin = half_fraction*abs(Jmin); None gives equal mass."""
    if half_fraction is None:
        return np.full(len(sample), 1 / len(sample))
    best = sample.analysis_score.min()
    delta = absolute_delta if absolute_delta is not None else half_fraction * abs(best)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('Zero minimum requires a positive ABSOLUTE_HALF_DELTA.')
    weight = np.exp(-np.log(2) * (sample.analysis_score.to_numpy() - best) / delta)
    return weight / weight.sum()


def weighted_summary(sample, name, reference, weights):
    """Inverse weighted empirical CDF quantiles in saved physical units."""
    values = parameter_values(sample, name, reference)
    positions = sample.index.get_indexer(values.index)
    weight = np.asarray(weights)[positions]
    if not len(values) or weight.sum() <= 0:
        return dict(summary='no valid values', n_valid=0, n_unique=0, ess=0.,
                    median_coordinate=np.nan, width_coordinate=np.nan, boundary_fraction=np.nan,
                    dominant=None, categories='{}')
    weight = weight / weight.sum()
    n_unique = sample.loc[values.index, 'parameter_hash'].nunique()
    meta = dict(n_valid=len(values), n_unique=n_unique, ess=1 / np.sum(weight**2))
    if reference['kind'] == 'categorical':
        frequencies = pd.Series(weight, index=values.to_numpy()).groupby(level=0).sum()
        return dict(meta, summary='; '.join(f'{k}: {v:.0%}' for k, v in frequencies.items()),
                    categories=json.dumps(frequencies.to_dict()), dominant=frequencies.idxmax())
    order = np.argsort(values.to_numpy(), kind='stable')
    cdf = np.cumsum(weight[order])
    coordinate = values.to_numpy()[order]
    quantiles = coordinate[np.minimum(np.searchsorted(cdf, [.1, .5, .9]), len(cdf)-1)]
    physical = 10**quantiles if reference['kind'] == 'log10' else quantiles
    span = reference['upper'] - reference['lower']
    boundary = ((values <= reference['lower'] + .02 * span) |
                (values >= reference['upper'] - .02 * span))
    return dict(meta, summary=f'{physical[0]:.4g} – {physical[2]:.4g} (median {physical[1]:.4g})',
                q10=physical[0], median=physical[1], q90=physical[2],
                median_coordinate=quantiles[1], width_coordinate=quantiles[2]-quantiles[0],
                boundary_fraction=float(weight @ boundary.to_numpy()),
                outside_fraction=float(weight @ ((values < reference['lower']) |
                                                (values > reference['upper'])).to_numpy()))


def robustness_table(pool, groups, references, fractions=(.3, .4, .5),
                     half_fractions=(None, .1, .25, .5), absolute_half_delta=None):
    rows = []
    scores = score_matrix(pool, {'objective': groups}).objective
    for counting in ('evaluations', 'configurations'):
        for fraction in fractions:
            sample, cutoff = analysis_sample(pool, scores, fraction, counting)
            for half in half_fractions:
                settings = dict(counting=counting, fraction=fraction,
                                half_fraction='equal' if half is None else str(half),
                                n_selected=len(sample), cutoff=cutoff,
                                best_score=float(sample.analysis_score.min()),
                                half_weight_delta=(np.nan if half is None else
                                    absolute_half_delta if absolute_half_delta is not None else
                                    half * abs(sample.analysis_score.min())))
                try:
                    weights = sample_weights(sample, half, absolute_half_delta)
                except ValueError as error:
                    rows.extend(dict(settings, parameter=name, status=str(error)) for name in references)
                    continue
                for name, reference in references.items():
                    rows.append(dict(settings, parameter=name, status='ok',
                                     **weighted_summary(sample, name, reference, weights)))
    return pd.DataFrame(rows)


def addition_evidence(candidates, records, objectives, references, fractions=(.3, .4, .5),
                      min_unique=20):
    """SED additions and leave-one-out comparisons on each UNION-compatible pool."""
    contrasts = [(f'All minus {group}', 'All', group,
                  [g for g in GROUPS if g != group], list(GROUPS)) for group in GROUPS]
    contrasts += [('SED', 'SED+' + group, group, ['SED'], ['SED', group])
                  for group in ('INT', 'PDI', 'ALMA')]
    effects, pairs, coverage = [], [], []
    for before, after, added, old_groups, new_groups in contrasts:
        partitions = available_score_pools(candidates, records, new_groups)
        if not partitions:
            coverage.append(dict(before=before, after=after, score_pool_key='unavailable', n_pool=0))
        for key, pool in partitions.items():
            coverage.append(dict(before=before, after=after, score_pool_key=key, n_pool=len(pool)))
            scores = score_matrix(pool, {'before': old_groups, 'after': new_groups})
            old_id, old_ties, _ = winner(pool, old_groups)
            new_id, new_ties, _ = winner(pool, new_groups)
            pairs.append(dict(before=before, after=after, added=added, score_pool_key=key,
                              n_pool=len(pool), before_id=old_id, after_id=new_id,
                              before_ties=old_ties, after_ties=new_ties))
            for fraction in fractions:
                old, _ = analysis_sample(pool, scores.before, fraction)
                new, _ = analysis_sample(pool, scores.after, fraction)
                for name, reference in references.items():
                    effect = distribution_effect(old, new, name, reference)
                    old_valid = parameter_values(old, name, reference).index
                    new_valid = parameter_values(new, name, reference).index
                    unique = min(old.loc[old_valid].parameter_hash.nunique(),
                                 new.loc[new_valid].parameter_hash.nunique())
                    effects.append(dict(before=before, after=after, added=added,
                        score_pool_key=key, fraction=fraction, parameter=name,
                        min_unique=unique, supported=unique >= min_unique, **effect))
    return pd.DataFrame(effects), pd.DataFrame(pairs), pd.DataFrame(coverage)


def evidence_summary(robustness, effects, references, family_members=None,
                     baseline_fraction=.4, min_unique=20):
    """One readable row per input; the long tables retain all numbers and pools."""
    rows = []
    for name, reference in references.items():
        data = robustness.loc[robustness.parameter.eq(name) & robustness.status.eq('ok')]
        baseline = data.loc[data.counting.eq('evaluations') & data.fraction.eq(baseline_fraction)
                            & data.half_fraction.eq('equal')]
        if baseline.empty:
            continue
        baseline = baseline.iloc[0]
        changes = effects.loc[effects.parameter.eq(name) & effects.supported] if not effects.empty else pd.DataFrame()
        shift_text, narrow_text = [], []
        if not changes.empty:
            # Keep scoring-pool identity; don't combine incompatible effects.
            for (before, after, key), group in changes.groupby(['before', 'after', 'score_pool_key']):
                at_baseline = group.loc[group.fraction.eq(baseline_fraction)]
                if at_baseline.empty:
                    continue
                row = at_baseline.iloc[0]
                label = f'{before} → {after} [{key[:8]}]'
                shift_text.append((row['shift'], f'{label}: {row["shift"]:.2g}'))
                if np.isfinite(row.width_ratio) and row.width_ratio < 1:
                    stable = len(group) == effects.fraction.nunique() and group.width_ratio.lt(1).all()
                    narrow_text.append((row.width_ratio, f'{label}: ×{row.width_ratio:.2g}'
                                        + ('; all cutoffs' if stable else '; cutoff sensitive')))
        families = []
        if family_members is not None and not family_members.empty:
            for family, members in family_members.groupby('family'):
                best = members.sort_values(['objective_score', 'evaluation_id']).iloc[0]
                families.append(f'F{family}: {best["param__" + name]}')
        if reference['kind'] == 'categorical':
            stability = 'Dominant categories: ' + ', '.join(sorted(data.dominant.dropna().unique()))
            boundary = 'categorical; see frequencies'
        else:
            span = reference['upper'] - reference['lower']
            shift = (data.median_coordinate.max() - data.median_coordinate.min()) / span if span > 0 else np.nan
            stability = (f'Median span: {shift:.1%} of allowed coordinate span; '
                         f'10–90% width: {data.width_coordinate.min()/span:.1%}–'
                         f'{data.width_coordinate.max()/span:.1%} of span' if span > 0 else 'fixed numeric bounds')
            boundary = f'{baseline.boundary_fraction:.0%} within outer 2% of bounds'
        rows.append(dict(parameter=name, preferred_sampled_range=baseline.summary,
            support=f'{int(baseline.n_unique)} unique / {int(baseline.n_valid)} evaluations'
                    + ('; sparse' if baseline.n_unique < min_unique else ''),
            largest_shifts=' | '.join(text for _, text in sorted(shift_text, reverse=True)[:2]) or 'insufficient compatible evidence',
            strongest_narrowing=('not defined for categories' if reference['kind'] == 'categorical' else
                ' | '.join(text for _, text in sorted(narrow_text)[:2]) or 'none supported'),
            boundary=boundary, robustness=stability,
            sampled_family_values=' | '.join(families) or 'not assessed'))
    return pd.DataFrame(rows)


def pair_changes(pair, candidates, records, references):
    """Physical parameter changes and every available original score contribution."""
    data = candidates.set_index('evaluation_id')
    before, after = data.loc[pair.before_id], data.loc[pair.after_id]
    parameters = pd.DataFrame([dict(parameter=name, before=before['param__' + name],
                                    after=after['param__' + name]) for name in references])
    scores = []
    pair_rows = candidates.loc[candidates.evaluation_id.isin([pair.before_id, pair.after_id])]
    required_ids = {pair.before_id, pair.after_id}
    for group, columns in GROUPS.items():
        partitions = available_score_pools(pair_rows, records, [group])
        comparable = any(required_ids <= set(frame.evaluation_id) for frame in partitions.values())
        for column in columns:
            old, new = before.get(column, np.nan), after.get(column, np.nan)
            scores.append(dict(observation=group, term=column, before=old, after=new,
                               comparable=comparable, change=new-old if comparable else np.nan))
    selected = candidates.loc[candidates.evaluation_id.isin([pair.before_id, pair.after_id])]
    pdi, alma = prediction_tables(selected, records)
    return parameters, pd.DataFrame(scores), pdi, alma


def plot_prediction_pair(pair, records, sed_tables):
    """Show changed SED residuals, PDI amplitudes/shapes and absolute ALMA profiles.

    Each model's own saved observation is drawn. Shapes are never renormalised
    here; PDI profiles were normalized by the original scoring routine.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for model_id, label, color in [(pair.before_id, 'Before', '#386cb0'),
                                   (pair.after_id, 'After', '#c14953')]:
        points = sed_tables.get(model_id)
        if points is not None:
            axes[0, 0].scatter(points.wavelength_um, points.residual_sigma, s=18,
                               alpha=.7, color=color, label=label)
        info = records[model_id]['info']
        for i, band in enumerate('VIH'):
            constraints = info.get('pdi', {}).get(band, {}).get('constraints', {})
            observed, model = constraints.get('observed', {}), constraints.get('model', {})
            obs_fraction = observed.get('positive_qphi_over_i', np.nan)
            if obs_fraction > 0:
                axes[0, 1].scatter([i], [model.get('positive_qphi_over_i', np.nan) / obs_fraction],
                                   color=color, label=label if i == 0 else None)
            edges = np.asarray(observed.get('radial_edges_mas', []))
            x = (edges[:-1] + edges[1:]) / 2
            prediction, observation = model.get('qphi_profile', []), observed.get('qphi_profile', [])
            ax = axes[1, i]
            if len(x) and len(x) == len(prediction) == len(observation):
                ax.plot(x, prediction, color=color, label=label)
                ax.plot(x, observation, '--', color=color, alpha=.6)
            ax.set(xlabel='Radius [mas]', ylabel='Saved normalized Qφ contribution', title=f'PDI {band}')
        alma = info.get('alma', {}).get('comparison', {})
        x = alma.get('radius_mas', [])
        predicted, observed = alma.get('model_profile_jybeam', []), alma.get('observed_profile_jybeam', [])
        if len(x) and len(x) == len(predicted) == len(observed):
            axes[0, 2].plot(x, predicted, color=color, label=label)
            axes[0, 2].plot(x, observed, '--', color=color, alpha=.6)
    axes[0, 0].axhline(0, color='grey', lw=1)
    axes[0, 0].set(xscale='log', xlabel='Wavelength [µm]', ylabel='(data − model) / σ', title='SED residuals')
    axes[0, 1].axhline(1, color='grey', lw=1)
    axes[0, 1].set(xticks=range(3), xticklabels=list('VIH'), ylabel='Model / observed polarized fraction', title='PDI amplitude')
    axes[0, 2].set(xlabel='Radius [mas]', ylabel='Intensity [Jy/beam]', title='ALMA absolute profile')
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, 'Prediction unavailable', ha='center', transform=ax.transAxes)
    fig.suptitle(f'{pair.before} → {pair.after}; pool {pair.score_pool_key[:8]}\n'
                 'Solid: model; dashed: corresponding observation. See score changes and image panels below.')
    return fig


def plot_cross_pairs(pool, groups, references, pairs, fraction=.4):
    """Selected 40% only, colored by unchanged objective score; no black overplot."""
    import matplotlib.pyplot as plt
    scores = score_matrix(pool, {'objective': groups}).objective
    selected, _ = analysis_sample(pool, scores, fraction)
    valid = [(x, y) for x, y in pairs if x in references and y in references]
    skipped = [(x, y) for x, y in pairs if (x, y) not in valid]
    if not valid:
        return None, skipped
    fig, axes = plt.subplots(1, len(valid), figsize=(5 * len(valid), 4), squeeze=False,
                             constrained_layout=True)
    for ax, (x, y) in zip(axes.flat, valid):
        coords = []
        for axis, name in [(ax.xaxis, x), (ax.yaxis, y)]:
            ref = references[name]
            values = parameter_values(selected, name, ref).reindex(selected.index)
            if ref['kind'] == 'categorical':
                values = values.map({v: i for i, v in enumerate(ref['categories'])})
                axis.set_ticks(range(len(ref['categories'])))
                axis.set_ticklabels(ref['categories'])
            coords.append(values)
        scatter = ax.scatter(*coords, c=selected.analysis_score, cmap='magma', s=24,
                             alpha=.8, edgecolors='none')
        def label(name):
            return ('log₁₀ ' if references[name]['kind'] == 'log10' else '') + parameter_label(name)
        ax.set(xlabel=label(x), ylabel=label(y))
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Original summed objective (lower is better)')
    fig.suptitle(f'Best {fraction:.0%} by count, including cutoff ties; {len(selected)} evaluations')
    return fig, skipped
