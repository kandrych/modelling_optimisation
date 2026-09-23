"""Named observable stages, independent of the physical ConfigSpace."""
import math
from pathlib import Path

import yaml


def load_fidelity_config(path):
    with Path(path).open() as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict) or set(settings) - {'eta', 'fidelity_stages'}:
        raise ValueError('Fidelity file must contain fidelity_stages and optional eta.')
    eta = settings.get('eta', 3)
    if isinstance(eta, bool) or not isinstance(eta, int) or eta < 2:
        raise ValueError('eta must be an integer >= 2.')
    stages = settings.get('fidelity_stages')
    if not isinstance(stages, list) or not stages:
        raise ValueError('fidelity_stages must be a nonempty list.')
    allowed = {'sed', 'pdi_V', 'pdi_I', 'pdi_H', 'alma', 'vis2_1perband', 'vis2_chromatic'}
    names, resolved = set(), []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or set(stage) != {'name', 'products'}:
            raise ValueError('Each fidelity stage requires only name and products.')
        name, products = stage['name'], stage['products']
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('Stage names must be nonempty and unique.')
        if (not isinstance(products, list) or not products
                or any(not isinstance(p, str) or p not in allowed for p in products)
                or len(set(products)) != len(products)):
            raise ValueError(f'Invalid or duplicate products in stage {name}.')
        if {'vis2_1perband', 'vis2_chromatic'} <= set(products):
            raise ValueError('Choose one interferometry mode per stage.')
        names.add(name)
        resolved.append(dict(stage=name, products=list(products), budget=float(eta**index), image_res=2))
    return {'eta': eta, 'stages': resolved}


def map_budget_to_fidelity(budget, settings):
    for stage in settings['stages']:
        if math.isclose(float(budget), stage['budget'], rel_tol=1e-9, abs_tol=0):
            return dict(stage, products=list(stage['products']))
    raise ValueError(f'Budget {budget} is not a configured fidelity level.')


def fidelity_products(settings):
    return list(dict.fromkeys(p for stage in settings['stages'] for p in stage['products']))
