"""Plot notation from paper/seed1_physical_model_and_optimisation.tex.

Keep saved configuration keys in data and exports; apply these labels only when
rendering figures. Matplotlib mathtext does not require a LaTeX installation.
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


def parameter_label(name):
    """Return the paper symbol, or the original label for unknown parameters."""
    return PARAMETER_LABELS.get(name, name)
