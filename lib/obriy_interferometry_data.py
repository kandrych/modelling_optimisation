"""Load each interferometric fidelity's existing observational selection separately."""
from pathlib import Path


def load_interferometry_data(data_root, products, reader):
    """Return mode -> instrument -> OIContainer; wavelengths are in microns.

    ``reader`` is Distroi's OIFITS reader. Separate reads preserve its existing
    wavelength/visibility filtering without modifying observations in place.
    """
    modes = [mode for mode in ("vis2_1perband", "vis2_chromatic") if mode in products]
    if not modes:
        return {}
    roots = {
        "demo_mac": "/Users/katerynaandrych/Work/lin/Postdoc/Data/interferometry/IRAS08544-4431",
        "demo_ozstar": "/fred/oz061/kandrych/Data/interferometry/IRAS08544-4431",
    }
    if data_root not in roots:
        raise ValueError(f"No interferometric observations configured for {data_root!r}.")
    instruments = {
        "pionier": ("PIONIER", "*.fits"),
        "gravity": ("GRAVITY", "*1.fits"),
        "matisse_l": ("MATISSE_L", "*.fits"),
        "matisse_n": ("MATISSE_N", "*.fits"),
    }
    selections = {
        "vis2_1perband": {
            "pionier": {"wave_lims": (1.63, 1.64)},
            "gravity": {"wave_lims": (2.199, 2.201)},
            "matisse_l": {"wave_lims": (3.48, 3.52)},
            "matisse_n": {"wave_lims": (9.9, 10.10), "fcorr": True},
        },
        "vis2_chromatic": {
            "pionier": {},
            "gravity": {},
            "matisse_l": {"wave_lims": (2.95, 3.95), "v2lim": 1e-8},
            "matisse_n": {"wave_lims": (8.0, 13.0), "v2lim": 1e-8, "fcorr": True},
        },
    }
    result = {}
    for mode in modes:
        result[mode] = {}
        for instrument, (folder, pattern) in instruments.items():
            try:
                result[mode][instrument] = reader(
                    str(Path(roots[data_root]) / folder) + "/", pattern,
                    **selections[mode][instrument])
            except Exception as error:
                raise RuntimeError(f"Failed to load {mode} observations for {instrument}.") from error
    return result
