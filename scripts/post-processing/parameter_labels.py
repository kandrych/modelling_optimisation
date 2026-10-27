"""Display notation and short meanings for recognised model inputs.

Keep saved configuration keys in data and exports; apply these labels only when
rendering figures. Bounds, types and ties come from each run's saved ConfigSpace;
this glossary does not prescribe a model setup. Matplotlib uses mathtext.
"""

PARAMETER_LABELS = {
    "disk_total_dust_mass": r"$M_{\mathrm{dust}}$",
    "zone_1_Rin": r"$R_{\mathrm{in}}$",
    "disk_Rmid": r"$R_{\mathrm{mid}}$",
    "zone_2_Rc": r"$R_c$",
    "zone_2_Rout": r"$R_{\mathrm{out}}$",
    "zone_1_surface_density_exp": r"$p_1$",
    "zone_2_surface_density_exp": r"$p_2$",
    "inclination": r"$i$",
    "disk_pa": r"$\mathrm{PA}$",
    "zone_1_scale_height": r"$H_0$",
    "zone_1_flaring_exp": r"$\beta$",
    "alpha_viscosity": r"$\alpha$",
    "zone_1_species_1_amin": r"$a_{\min}$",
    "zone_1_species_1_amax": r"$a_{\max}$",
    "zone_1_species_1_aexp": r"$q$",
    "zone_1_species_1_Vmax": r"$V_{\max}$",
    "zone_1_species_1_porosity": r"$\mathcal{P}$",
    "zone_1_species_1_component_2_volume_fraction_iron": r"$f_{\mathrm{Fe}}$",
    "zone_2_-gamma_exp": r"$p_3$"
}


PARAMETER_DESCRIPTIONS = {
    "disk_total_dust_mass": "Combined dust mass [solar masses]; zone masses are derived in the continuous two-zone setup.",
    "zone_1_Rin": "Inner radius of zone 1 [au].",
    "disk_Rmid": "Joining radius of the two zones [au] in the continuous two-zone setup.",
    "zone_2_Rc": "Characteristic taper radius of zone 2 [au].",
    "zone_2_Rout": "Outer radius of zone 2 [au].",
    "zone_1_surface_density_exp": "Saved signed radial surface-density exponent in zone 1.",
    "zone_2_surface_density_exp": "Saved signed radial surface-density exponent in zone 2.",
    "inclination": "Viewing inclination [degrees]; the writer sets imin and imax to this value.",
    "disk_pa": "Disc position angle [degrees], using the run's orientation convention.",
    "zone_1_scale_height": "Scale height [au] at zone_1_Rref; the reference radius is setup-dependent.",
    "zone_1_flaring_exp": "Exponent controlling the radial change of scale height.",
    "alpha_viscosity": "Dimensionless viscosity parameter; affects settling/heating when those options are enabled.",
    "zone_1_species_1_amin": "Minimum grain radius [micrometres].",
    "zone_1_species_1_amax": "Maximum grain radius [micrometres].",
    "zone_1_species_1_aexp": "Exponent of the grain-size distribution.",
    "zone_1_species_1_Vmax": "Maximum DHS void fraction; distinct from material porosity.",
    "zone_1_species_1_porosity": "Material porosity fraction.",
    "zone_1_species_1_component_2_volume_fraction_iron": "Iron volume fraction in the composite grain population.",
    "zone_2_-gamma_exp": "Saved signed taper parameter; taper power is 2 + this value. May be tied to the outer density slope.",
}


def parameter_description(name):
    """Explain a recognised input without inferring unfamiliar setup conventions."""
    return PARAMETER_DESCRIPTIONS.get(name, "See this run's configuration and parameter-file writer.")


def parameter_label(name):
    """Return the paper symbol, or the original label for unknown parameters."""
    return PARAMETER_LABELS.get(name, name)
