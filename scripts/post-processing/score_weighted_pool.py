"""Descriptive Gibbs weights on saved View-A evaluations, not posterior inference.

The base measure gives every evaluation equal mass, including repeats. Nothing
here corrects the adaptive SMAC sampling density or changes an original score.
ChainConsumer is optional until plotting; the weighting tables need no new deps.
"""
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

from model_analysis_utils import score_matrix
from observation_convergence_utils import parameter_values
from parameter_labels import parameter_label


# These classify independent inputs; they never add fixed or derived quantities.
# In seed1, inclination drives imin/imax, Rmid drives the shared boundary,
# total mass drives zone masses, and zone-1 grain properties are shared.
PHYSICAL_PARAMETER_GROUPS = {
    'Disc geometry and orientation': [
        'inclination', 'disk_pa', 'disk_Rmid', 'imin', 'imax', 'n_incl',
        'centered', 'az_min', 'az_max', 'distance_pc',
        *[f'zone_{zone}_{suffix}' for zone in (1, 2) for suffix in (
            'Rin', 'Rout', 'Rc', 'Rref', 'edge', 'scale_height',
            'vertical_profile_exponent', 'flaring_exp', 'surface_density_exp', '-gamma_exp')]],
    'Dust distribution, settling and heating': [
        'disk_total_dust_mass', 'alpha_viscosity', 'dust_settling', 'exp_strat',
        'a_strat', 'dust_radial_migration', 'sublimate_dust',
        'hydrostatic_equilibrium', 'viscous_heating',
        *[f'zone_{zone}_{suffix}' for zone in (1, 2) for suffix in ('dust_mass', 'gas_to_dust')],
        *[f'zone_{zone}_species_1_{suffix}' for zone in (1, 2) for suffix in (
            'amin', 'amax', 'aexp', 'Vmax', 'porosity',
            'component_2_volume_fraction_iron', 'component_2_volume_fraction_forsterite',
            'grain_type', 'N_components', 'mixing_rule', 'mass_fraction',
            'optical_indices_file', 'volume_fraction', 'heating_method', 'n_grains')]],
    'Stellar properties': [f'star_1_{suffix}' for suffix in (
        'Temp', 'R', 'M', 'x', 'y', 'z', 'autometic_spectrum', 'spectrum_file', 'fUV', 'slope_fUV')],
    'Numerical grid': ['grid_geometry', 'n_rad', 'nz', 'n_az', 'n_rad_in', 'grid_nx', 'grid_ny', 'grid_size'],
}


def group_parameter_references(references, groups=None):
    """Partition supplied independent references once; retain unknowns explicitly.

    Custom groups replace the defaults. Unassigned parameters go to Other
    parameters, while duplicate assignments raise instead of duplicating axes.
    Parameter types and log coordinates come from the pool, not this taxonomy.
    """
    groups = PHYSICAL_PARAMETER_GROUPS if groups is None else groups
    result, assigned = {}, set()
    for label, names in groups.items():
        selected = {}
        for name in names:
            if name not in references:
                continue
            if name in assigned:
                raise ValueError(f'Parameter assigned to multiple groups: {name}')
            assigned.add(name)
            selected[name] = references[name]
        if selected:
            result[label] = selected
    remaining = {name: ref for name, ref in references.items() if name not in assigned}
    if remaining:
        result.setdefault('Other parameters', {}).update(remaining)
    return result


def score_weighted_pool(pool, objective, groups, half_weight_fractions=(.1, .25, .5),
                        absolute_half_weight_deltas=None):
    """Return per-evaluation weights and temperature diagnostics for one pool.

    A score increase delta halves the weight: T = delta / log(2). Absolute
    deltas, if supplied, replace fractional deltas for this entire call. Relative
    deltas are fraction * abs(best), never percentiles of the sampled scores.
    A zero best requires explicit positive absolute deltas; no fallback scale
    is invented. There is no top-N cut, repeat aggregation or prior correction.
    """
    if pool.empty or pool.score_pool_key.nunique(dropna=False) != 1:
        raise ValueError('Supply one nonempty compatible score pool.')
    if pool.score_pool_key.isna().any() or not pool.evaluation_id.is_unique:
        raise ValueError('A known score-pool key and unique evaluation IDs are required.')
    if 'physical_pool_key' in pool and pool.physical_pool_key.nunique(dropna=False) != 1:
        raise ValueError('Do not merge physical pools.')
    if pool.parameter_hash.isna().any():
        raise ValueError('Configuration fingerprints are required for repeat diagnostics.')
    scores = score_matrix(pool, {objective: groups})[objective].to_numpy(float)
    if not np.isfinite(scores).all():
        raise ValueError('The summed objective must be finite.')
    best = float(scores.min())
    absolute = absolute_half_weight_deltas is not None
    values = np.asarray(absolute_half_weight_deltas if absolute else half_weight_fractions,
                        dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Half-weight deltas/fractions must be a nonempty list of finite positive values.')
    if len(np.unique(values)) != len(values):
        raise ValueError('Half-weight settings must be distinct.')
    if not absolute and best == 0:
        raise ValueError('Best score is zero: set positive absolute half-weight deltas for this pool.')
    deltas = values if absolute else values * abs(best)
    temperatures = deltas / np.log(2.)
    if not np.isfinite(temperatures).all() or (temperatures <= 0).any():
        raise ValueError('Temperature must be finite and positive in original score units.')
    metadata_columns = [c for c in ('evaluation_id', 'parameter_hash', 'history',
        'physical_pool_key', 'score_pool_key') if c in pool]
    parts, diagnostics = [], []
    for index, (value, delta, temperature) in enumerate(zip(values, deltas, temperatures), 1):
        # Long double subtraction avoids overflow for widely separated finite scores.
        with np.errstate(over='ignore', under='ignore'):
            log_relative = np.asarray(-(scores.astype(np.longdouble)-best)/temperature, dtype=float)
            log_weight = log_relative-logsumexp(log_relative)
            weight = np.exp(log_weight)
        setting = f'half_weight_{index:02d}'
        label = (f'half-weight at +{delta:.4g} score' if absolute else
                 f'half-weight at +{value:.0%} |best|')
        rows = pool[metadata_columns].copy().reset_index(drop=True)
        rows = rows.assign(objective=objective, setting=setting, label=label,
            objective_score=scores, log_relative_weight=log_relative,
            log_weight=log_weight, weight=weight, temperature=float(temperature),
            half_weight_delta=float(delta))
        parts.append(rows)
        configuration_mass = rows.groupby('parameter_hash').weight.sum()
        diagnostics.append(dict(objective=objective, score_pool_key=pool.score_pool_key.iloc[0],
            setting=setting, label=label, temperature=float(temperature),
            half_weight_delta=float(delta), best_score=best,
            half_weight_fraction=float(value) if not absolute else np.nan,
            temperature_rule='absolute_score_delta' if absolute else 'fraction_of_abs_best',
            base_measure='equal_per_evaluation_including_repeats', n_evaluations=len(pool),
            n_configurations=pool.parameter_hash.nunique(), n_histories=pool.history.nunique(),
            n_zero_weights=int((weight == 0).sum()), max_evaluation_weight=float(weight.max()),
            evaluation_weight_ess=float(1./np.sum(weight**2)),
            configuration_weight_ess=float(1./np.sum(configuration_mass.to_numpy()**2))))
    return pd.concat(parts, ignore_index=True), pd.DataFrame(diagnostics)


def plot_score_weighted_pool(pool, weights, diagnostics, references,
                             contour_masses=(.68, .95), bins=20, smooth=0,
                             min_weight_ess=20, style='sensitivity', reference_setting=None,
                             color_scale='auto', parameter_group=None, complete_references=None):
    """Plot temperature overlays or a single reference-style score cloud.

    Reference style uses black contours, descriptive multimodal HDI marginal
    summaries, and a star/reference lines for the best displayed evaluation.
    Colour encodes the original score; the point cloud is not weight-resampled.

    All numeric parameters share a complete-case subset, reported with its lost
    weight. Categories use separate exact weighted bars (no numeric encoding).
    Constant numeric parameters are recorded, not given artificial contours.
    For grouped plots, complete_references supplies the full parameter set so
    every group uses the same numeric complete cases, including category bars.
    Categories additionally require their own value to be present.
    Suppress corner curves with concentrated weights; ESS is a concentration
    diagnostic only, never an independent-sample count or convergence test.
    """
    from chainconsumer import Chain, ChainConsumer, PlotConfig, Truth
    from chainconsumer.statistics import SummaryStatistic
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

    masses = np.asarray(contour_masses, dtype=float)
    if masses.ndim != 1 or not len(masses) or not np.isfinite(masses).all() or ((masses <= 0) | (masses >= 1)).any():
        raise ValueError('Contour masses must lie strictly between zero and one.')
    if bins < 2 or smooth < 0 or not np.isfinite(min_weight_ess) or min_weight_ess < 1:
        raise ValueError('Use bins >= 2, smooth >= 0 and minimum weight ESS >= 1.')
    if pool.empty or not pool.evaluation_id.is_unique or pool.score_pool_key.nunique() != 1:
        raise ValueError('Supply one nonempty score pool with unique evaluation IDs.')
    if diagnostics.empty or diagnostics.objective.nunique() != 1 or diagnostics.score_pool_key.nunique() != 1:
        raise ValueError('Supply weights/diagnostics for one objective and score pool.')
    if diagnostics.score_pool_key.iloc[0] != pool.score_pool_key.iloc[0]:
        raise ValueError('Weight diagnostics do not belong to this score pool.')
    if style not in {'reference', 'sensitivity'}:
        raise ValueError('Use reference or sensitivity plot style.')
    if color_scale not in {'auto', 'log', 'linear'}:
        raise ValueError('Use auto, log or linear score colour scaling.')
    if style == 'reference':
        if reference_setting is None:
            reference_setting = diagnostics.setting.iloc[len(diagnostics)//2]
        diagnostics = diagnostics.loc[diagnostics.setting.eq(reference_setting)]
        if len(diagnostics) != 1:
            raise ValueError('Reference style requires one existing weighting setting.')
    frame = pool.set_index('evaluation_id', drop=False)
    numeric, categorical, coverage = {}, {}, []
    for name, ref in references.items():
        values = parameter_values(frame, name, ref).reindex(frame.index)
        if ref['kind'] == 'categorical':
            categorical[name] = values
        elif values.nunique() > 1:
            numeric[name] = values
        else:
            coverage.append(dict(parameter=name, status='constant_or_unavailable',
                                 n_valid=int(values.notna().sum())))
    coordinates = pd.DataFrame(numeric, index=frame.index)
    complete = coordinates.notna().all(axis=1)
    if complete_references is not None:
        for name, ref in complete_references.items():
            if ref['kind'] != 'categorical':
                values = parameter_values(frame, name, ref).reindex(frame.index)
                if values.nunique() > 1:
                    complete &= values.notna()
    labels = {name: parameter_label(name) + (' (log10)' if references[name]['kind'] == 'log10'
                                          else ' (saved units)') for name in numeric}
    if style == 'reference':
        # ChainConsumer embeds these labels inside mathematical summary titles.
        # Keep the entire label in one math block, including log coordinates.
        for name in numeric:
            symbol = parameter_label(name)
            symbol = symbol.strip('$') if symbol.startswith('$') else r'\mathrm{' + symbol.replace('_', r'\_') + '}'
            labels[name] = '$' + (r'\log_{10} ' if references[name]['kind'] == 'log10' else '') + symbol + '$'
    extents = {name: (float(values.min()), float(values.max())) for name, values in numeric.items()}
    consumer = ChainConsumer()
    figures = {}
    colours = plt.get_cmap('tab10')
    category_rows = []
    for i, row in enumerate(diagnostics.itertuples(index=False)):
        chosen = weights.loc[weights.setting.eq(row.setting)]
        if (len(chosen) != len(frame) or not chosen.evaluation_id.is_unique
                or set(chosen.evaluation_id) != set(frame.index)
                or not chosen.objective.eq(row.objective).all()
                or not chosen.score_pool_key.eq(row.score_pool_key).all()):
            raise ValueError('Weights must match every evaluation in this objective/pool exactly.')
        w = chosen.set_index('evaluation_id').weight.reindex(frame.index)
        if not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1.):
            raise ValueError('Weights must be finite, nonnegative and normalised.')
        valid = complete & w.gt(0)
        kept_mass = float(w.loc[valid].sum())
        ess = float(kept_mass**2 / (w.loc[valid]**2).sum()) if kept_mass else 0.
        config_weights = w.loc[valid].groupby(frame.loc[valid, 'parameter_hash']).sum()
        config_ess = float(kept_mass**2 / (config_weights**2).sum()) if kept_mass else 0.
        supported = (bool(numeric) and min(ess, config_ess) >= min_weight_ess
                     and coordinates.loc[valid].nunique().ge(2).all())
        coverage.append(dict(parameter='numeric_corner', setting=row.setting,
            status='plotted' if supported else 'insufficient_weight_support_or_no_varying_numeric_parameters',
            n_pool=len(frame), n_valid=int(valid.sum()), retained_weight=kept_mass,
            omitted_weight=max(0., 1.-kept_mass), evaluation_weight_ess=ess,
            configuration_weight_ess=config_ess))
        if supported:
            samples = coordinates.loc[valid].copy()
            samples['weight'] = w.loc[valid]/kept_mass
            options = {}
            if style == 'reference':
                selected_scores = chosen.set_index('evaluation_id').objective_score.reindex(samples.index)
                score_min, score_max = float(selected_scores.min()), float(selected_scores.max())
                if color_scale == 'log' and score_min <= 0:
                    raise ValueError('Log colour scaling requires strictly positive objective scores; use auto or linear.')
                actual_color_scale = ('log' if score_min > 0 and score_max > score_min
                                      and color_scale in {'auto', 'log'} else 'linear')
                score_norm = (LogNorm(vmin=score_min, vmax=score_max) if actual_color_scale == 'log'
                              else Normalize(vmin=score_min, vmax=score_max))
                # Original loss colours the cloud. The ranking column only locates
                # the best model; it is not a log likelihood or a physical posterior.
                samples['objective_score'] = selected_scores
                samples['_best_fit_rank'] = -selected_scores
                # Stable tie-breaking makes the star and dashed lines identify one
                # actual evaluated parameter vector, not independent marginal medians.
                samples = samples.sort_index().sort_values('objective_score', kind='stable')
                best_id = samples.index[0]
                coverage[-1].update(reference_evaluation_id=best_id,
                    reference_score=float(selected_scores.loc[best_id]),
                    n_tied_best=int(selected_scores.eq(selected_scores.min()).sum()),
                    score_color_scale=actual_color_scale, score_color_min=score_min,
                    score_color_max=score_max, score_colormap='magma[0.20:0.95]')
                consumer.add_truth(Truth(location=coordinates.loc[best_id].to_dict(),
                    name='Best displayed evaluation (not known truth)', color='#500724'))
                options = dict(posterior_column='_best_fit_rank', color_param='objective_score',
                    plot_cloud=True, num_cloud=len(samples), cmap='magma', plot_point=True,
                    marker_style='*', marker_size=100, linewidth=2. if len(numeric) <= 3 else .8,
                    summary_area=.68)
            # ChainConsumer 1.3's point locator requires positional integer labels.
            # Evaluation IDs remain in the exported weights and coverage tables.
            consumer.add_chain(Chain(samples=samples.reset_index(drop=True), name=f'{row.label}; T={row.temperature:.4g}',
                multimodal=True, statistics=SummaryStatistic.HDI,
                weight_column='weight', color='k' if style == 'reference' else colours(i % 10),
                bins=bins, smooth=smooth,
                kde=False, sigmas=[0., *norm.ppf((1.+np.sort(np.unique(masses)))/2.).tolist()],
                show_contour_labels=style == 'reference' and len(numeric) <= 3, shade=False,
                bar_shade=style == 'reference', **options))
        for name, values in categorical.items():
            available = values.notna()
            if complete_references is not None:
                available &= complete
            mass = float(w.loc[available].sum())
            fractions = w.loc[available].groupby(values.loc[available]).sum()
            for category in references[name]['categories']:
                category_rows.append(dict(parameter=name, setting=row.setting, label=row.label,
                    category=category, weight=float(fractions.get(str(category), 0.)/mass) if mass else np.nan,
                    retained_weight=mass, omitted_weight=max(0., 1.-mass),
                    n_valid=int(available.sum()), status='category_bar' if mass else 'no_category_weight'))
    selection_caption = ''
    if 'selection_fraction' in diagnostics:
        selection = diagnostics.iloc[0]
        selection_caption = (f'Best {selection.selection_fraction:.0%}: '
            f'{int(selection.n_evaluations)}/{int(selection.n_before_selection)} evaluations; '
            f'score ≤ {selection.selection_cutoff:.5g} (cutoff ties included)')
    if any(item.get('status') == 'plotted' for item in coverage):
        n_shown = sum(item.get('status') == 'plotted' for item in coverage)
        labels['objective_score'] = f'{diagnostics.objective.iloc[0]} score (lower is better)'
        if style == 'reference':
            labels['objective_score'] = (f'{diagnostics.objective.iloc[0]} score '
                f'({actual_color_scale} colour scale; lower is better)')
        consumer.set_plot_config(PlotConfig(labels=labels, extents=extents,
            summarise=style == 'reference', flip=style == 'reference' and len(numeric) > 1,
            contour_label_font_size=12, sigma2d=False, usetex=False, serif=False,
            show_legend=style != 'reference'))
        figure_size = (10, 6) if len(numeric) == 1 else (11, 10) if len(numeric) == 2 else None
        fig = consumer.plotter.plot(figsize=figure_size) if style == 'reference' else consumer.plotter.plot()
        if style == 'reference':
            # ChainConsumer forwards a colour column but not a Matplotlib norm.
            # Update only the score clouds; the attached colourbar follows their
            # mappable, keeping ticks in original score units. No scores are clipped.
            visible_magma = LinearSegmentedColormap.from_list(
                'visible_magma', plt.get_cmap('magma')(np.linspace(.20, .95, 256)))
            for ax in fig.axes:
                for collection in ax.collections:
                    if isinstance(collection, PathCollection) and collection.get_array() is not None:
                        collection.set_norm(score_norm)
                        collection.set_cmap(visible_magma)
        percentages = ', '.join(f'{100*mass:g}%' for mass in sorted(set(masses)))
        title = f'View A — {diagnostics.objective.iloc[0]} | pool {diagnostics.score_pool_key.iloc[0][:12]}\n'
        if parameter_group:
            title += parameter_group + '\n'
        if selection_caption:
            title += selection_caption + '\n'
        if style == 'reference':
            row = diagnostics.iloc[0]
            density_label = 'smoothed pool weight' if smooth else 'pool weight'
            title += f'{row.label}; T={row.temperature:.4g}'
            title += (f'; contours: {percentages} of {density_label}' if len(numeric) > 1
                      else f'; marginal: {density_label}')
            reference_caption = 'Star/dashes' if len(numeric) > 1 else 'Dashed line'
            fig.text(.5, .025,
                f'{reference_caption}: best displayed model. Marginals: 68% highest-density bands of plotted weight.\n'
                'Saved parameter units (log10 where labelled). Adaptive SMAC pool; not calibrated uncertainty.',
                ha='center', va='bottom', fontsize=9)
            if len(numeric) == 1:
                fig.axes[0].set_position((.12, .22, .83, .48))
            elif len(numeric) == 2:
                # Adjust all axes together: calling subplots_adjust after a colourbar
                # is made can move the right marginal over that colourbar.
                positions = [(.12, .68, .55, .17), (.695, .68, .15, .17),
                             (.12, .17, .55, .48), (.695, .17, .15, .48),
                             (.89, .17, .016, .68)]
                for ax, position in zip(fig.axes, positions):
                    ax.set_position(position)
            chain = consumer.get_chain(f'{row.label}; T={row.temperature:.4g}')
            summaries = consumer.analysis.get_summary(chains=[chain], columns=list(numeric))[chain.name]
            for name, summary in summaries.items():
                bands = summary if isinstance(summary, list) else [summary]
                for band_index, band in enumerate(bands, 1):
                    coverage.append(dict(parameter=name, setting=row.setting,
                        status='descriptive_marginal_summary', coordinates=references[name]['kind'],
                        lower=band.lower, center=band.center, upper=band.upper,
                        center_kind='band_mode' if len(bands) > 1 else 'within_interval_median',
                        summary_statistic='hdi', band_index=band_index, n_bands=len(bands),
                        summary_mass=.68, summary_mass_scope='all_bands_together'))
        else:
            title += (f'Score-weighted available evaluations; contours: {percentages} of plotted weight\n'
                f'{n_shown}/{len(diagnostics)} temperatures shown; adaptive SMAC pool; repeats retained; '
                'not calibrated uncertainty')
        fig.suptitle(title, fontsize=10)
        if style != 'reference':
            fig.subplots_adjust(top=.88 if len(numeric) < 4 else .94)
        figures['corner'] = fig
    if categorical:
        fig, axes = plt.subplots(len(categorical), 1, figsize=(9, 3.5*len(categorical)+1), squeeze=False)
        cat_table = pd.DataFrame(category_rows)
        width = .8/len(diagnostics)
        for ax, name in zip(axes.flat, categorical):
            categories = references[name]['categories']
            for i, row in enumerate(diagnostics.itertuples(index=False)):
                selected = cat_table.loc[cat_table.parameter.eq(name) & cat_table.setting.eq(row.setting)]
                y = selected.set_index('category').weight.reindex(categories).to_numpy()
                ax.bar(np.arange(len(categories))-.4+width*(i+.5), y, width=width,
                       label=f'{row.label}; T={row.temperature:.4g}', color=colours(i % 10))
            ax.set_xticks(range(len(categories)), categories)
            ax.set(xlabel=parameter_label(name), ylabel='Normalised available-model weight', ylim=(0, 1))
            ax.legend(fontsize=8)
        fig.suptitle(f'View A — {diagnostics.objective.iloc[0]} | pool {diagnostics.score_pool_key.iloc[0][:12]}\n'
                     + (parameter_group + '\n' if parameter_group else '')
                     + 'Categorical parameters: score-weighted evaluations; repeats retained'
                     + ('\n' + selection_caption if 'selection_fraction' in diagnostics else ''))
        fig.tight_layout(rect=(0, 0, 1, .92))
        figures['categories'] = fig
        coverage.extend(category_rows)
    report = pd.DataFrame(coverage).assign(objective=diagnostics.objective.iloc[0],
                                          score_pool_key=diagnostics.score_pool_key.iloc[0],
                                          plot_style=style, histogram_smooth=smooth,
                                          parameter_group=parameter_group)
    return figures, report
