# Two contiguous zones

## Optional scale-height tie

To derive zone 2's scale height from zone 1, use:

```yaml
- name: zone_2_scale_height
  type: constant
  value: "zone_1_scale_height"
```

Zone 1's scale height may be sampled, fixed in the configuration, or taken
from the template. It must be numeric, finite and positive. The writer
resolves the tie before writing each trial or final model, records the numeric
result in config_used.json, and leaves the sampled configuration unchanged.
No additional CLI flag is required. This ties only the heights at their
reference radii; identical vertical profiles also require matching reference
radii and flaring exponents. Only zone 2 may reference zone 1; the reverse tie is rejected.

## Optional shared dust population

Use `--share-zone-composition` to copy zone 1's entire dust population into
zone 2. It defaults to off and works independently of `--2ZONE_CONT_LHC`
and alongside `--tapered-edge-p1-eq-p2`. Exactly two template zones are required.

Zone 1 is the source after sampled grain parameters and conditional material
fractions have been applied. The copy includes species and component counts,
optical-index files, material volume fractions, species mass fractions,
grain type, mixing rule, porosity, Vmax, grain-size distributions and grids,
and heating methods. The generated file replaces zone 2's whole grain block,
even if its original species/component counts differ. The input template is
not modified. For example, a zone-2-only PAH species is removed when absent
from zone 1.

Remove `zone_2_number_of_species` and all `zone_2_species_*` entries from the
sampled configuration and associated conditions/forbiddens; the writer rejects
them when sharing is enabled. Sample dust properties only for zone 1.
The generated `config_used.json` records the effective dust parameters in both
zones. Zone dust masses, gas-to-dust ratios, radial geometry and scale-height
parameters are not copied. This changes the assumed dust populations; costs
and observational weights are unchanged.

## Continuous surface density

Enable with `python SMAC3.py ... --2ZONE_CONT_LHC`. The flag defaults to false.
The mode changes the physical parameter mapping; it does not change the SMAC
sampler or cost function. In particular, the name does not enable Latin
hypercube sampling.

In the run's actual YAML configuration, add `disk_Rmid` (au) and
`disk_total_dust_mass` (solar masses). Use the existing `uniform_float`,
`categorical`, or `constant` schema and choose physical bounds for the run.
For example, these **illustrative constants** reproduce a 10--100 au disc
split at 20 au, with total dust mass 0.009 solar masses:

```yaml
hyperparameters:
  - name: disk_Rmid
    type: constant
    value: 20.0
  - name: disk_total_dust_mass
    type: constant
    value: 0.009
  - name: zone_1_Rin
    type: constant
    value: 10.0
  - name: zone_2_Rout
    type: constant
    value: 100.0
  - name: zone_1_surface_density_exp
    type: constant
    value: -1.0
  - name: zone_2_surface_density_exp
    type: constant
    value: -1.0
conditions: []
forbiddens: []
```

Merge the needed entries into your existing run configuration. Remove
`zone_1_Rout`, `zone_2_Rin`, `zone_1_dust_mass`, and `zone_2_dust_mass`
from that configuration (and any conditions/forbiddens referencing them):
they are derived, not independently sampled. Their old template values are
overwritten. `disk_Rmid` is an absolute radius, not `Rmid/Rin`.

Supply a `simulation.para` template with exactly two zones. Zone 1 must have
density type 1; zone 2 may have type 1 or type 2. Both require `edge=0`.
The writer validates these assumptions rather than silently
changing them. Inner/outer radii and slopes use the configuration values when
provided, otherwise their template values. Require
`0 < zone_1_Rin < disk_Rmid < zone_2_Rout` for every sampled configuration.

The mass helper sets `zone_1_Rout = zone_2_Rin = disk_Rmid` and computes the
mass split from power-law radial integrals normalised at that boundary.
It handles slope -2 using the logarithmic integral. The example yields
zone masses 0.001 and 0.008 solar masses. Different slopes give a continuous
surface density but generally a kink. Scale heights and dust properties
remain independently configured.

## Optional tapered second zone

Keep the same `--2ZONE_CONT_LHC` flag. Set `zone_2_type` to `2` in the
template or run configuration. Supply `zone_2_Rc` (au) and
`zone_2_-gamma_exp` in the template or configuration; configuration values
take precedence. The latter is MCFOST's signed p2, not its negation.
Require finite `Rc > 0` and `p2 > -2` for an outward exponential taper.
Use an explicit positive `zone_2_Rout`; this mode does not accept MCFOST's
`Rout=0` automatic-radius convention.

For zone 2 the surface density is proportional to
`r**s2 * exp(-(r/Rc)**q)`, with `q = 2 + p2`. The mass integral is normalised
at `Rmid` and evaluated by quadrature:

```text
J2 = integral from 1 to Rout/Rmid of
     x**(s2+1) * exp(-(Rmid/Rc)**q * (x**q - 1)) dx
```

The code integrates in log radius for numerical stability and rejects
inaccurate quadrature results. The analytical power-law branch is unchanged.
This ensures matching surface densities, not matching slopes. A kink-free
join additionally requires `s1 = s2 - q*(Rmid/Rc)**q`; this is not imposed.

`--tapered-edge-p1-eq-p2` ties the two exponents in the last template zone:
zone 1 for a one-zone template, zone 2 for a two-zone template. It uses the
sampled surface-density exponent, or the template value if not sampled, and
overrides that zone's `-gamma_exp`. Remove that redundant exponent from the
sampled space when using this flag. The tie is applied before mass splitting
and is recorded in `config_used.json` for both trials and final models.
It does not change the zone's density type. For a tapered second zone in
`2ZONE_CONT_LHC` mode the shared exponent must still exceed -2.

Both trial and final-model `config_used.json` files record derived values.
SMAC's sampled configuration/run history retains the independent parameters.
Do not reuse old warm-start costs if their physical parameter mapping differs.
