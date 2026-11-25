# SpectralGrouping.jl

Detector-channel grouping, shared model grids, and spectrum combination for
OGIP observations loaded with SpectralFitting.jl. These are the numerical
tools for the Julia port of fancy_plotting.

## Install

Extract the package, then run the following from the directory that contains
`SpectralGrouping.jl/`. Add AstroRegistry once if it is missing from your Julia
installation.

```julia
import Pkg
Pkg.Registry.add("General")
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/astro-group-bristol/AstroRegistry"))
Pkg.develop(path="SpectralGrouping.jl")
```

The package targets Julia 1.10 or later and SpectralFitting 0.7.3 or 0.7.4.

## Group a dataset

Load the original PHA and apply any fit mask before grouping. Pass `background`,
`response`, or `ancillary` to `OGIPDataset` if the file paths need overrides.

```julia
using SpectralFitting, SpectralGrouping

data = OGIPDataset("source.pha")
mask_energies!(data, 0.5, 8.0)

by_counts = group_counts(data, 20)
by_width = group_width(data, 4)
by_snr = group_snr(data, 5.0; max_channels=30)
by_file_flags = group_ogip(data)
```

For RGS spectra loaded in descending energy order, call
`restore_detector_bounds!(data)` immediately after `OGIPDataset` and before
`mask_energies!`. This restores both RMF EBOUNDS for each channel by its label,
so energy cuts use the measured channel intervals. SpectralFitting may emit
its alignment warning during loading, before the helper runs. Reload spectra
with existing energy cuts to apply those cuts again using the restored bounds.

Each command returns a new `SpectralData`. Keep the original dataset to try
different groupings or return to native channels. Grouping by counts and S/N uses
observed source-region counts, including any background events in that region.
Apply those rules before background subtraction. S/N uses Poisson noise by
default. Supply `errors=data.spectrum.errors` to `group_snr` to use independent
errors from the spectrum.

Groups stop at masked or bad-quality source channels, missing channel labels,
and gaps in the RMF EBOUNDS. Short groups stay active in the result. The same
groups sum the source, the background matched by channel label, and the
detector rows of the RMF. The incident model grid and ARF retain their original
resolution. Numeric statistical errors are added in quadrature. Poisson
errors use SpectralFitting's count error at the summed counts.

Pass the result to `FittingProblem(model => by_counts)` to fit on the grouped
detector grid. With SpectralFitting 0.7.x, for chi-squared fits you can store
the grouped data in the statistic's preferred units first. This also converts
the stored one-sigma errors before SpectralFitting squares them.

```julia
set_units!(by_counts, preferred_units(by_counts, ChiSquared()))
```

## Reuse the grouping for plots

Use a plan when you need the native channel slices for plotting or want to
check which bins met the target.

```julia
plan = grouping_plan(data, :counts, 20)
grouped = grouped_dataset(data, plan)
short = findall(.!plan.met_target)
lo, hi = grouped.energy_low, grouped.energy_high

# folded_native_counts is a vector on the original detector channels.
folded_grouped_counts = group_sum(plan, folded_native_counts)
```

Use the paired `energy_low` and `energy_high` arrays for plotting across detector
gaps. `group_sum` expects integrated values in the source's original channel
order, including rows outside the selected range.

## Match and fit multiple spectra

For simultaneous W-stat fits with more than one incident grid, see the
[multi-grid W-stat guide](docs/multi-grid-wstat.md).

Load each observation with its response and set the detector masks first. Build
the shared incident grid before choosing a model. The grid contains all native
RMF input boundaries in keV. Supply `grid=` to `match_dataset_grids` to use
a contiguous custom edge vector. It must include every native bin boundary.

```julia
using SpectralFitting, SpectralGrouping

heg = OGIPDataset("heg.pha")
meg = OGIPDataset("meg.pha")
matched = match_dataset_grids(heg, meg; reference=1)

plans = (grouping_plan(heg, :counts, 15), grouping_plan(meg, :counts, 15))
joint = combine_datasets(matched; groupings=plans)
source = PowerLaw()
problem = FittingProblem(source => joint)
stat = grid_statistic(joint)
```

This fits one source expression. Each objective evaluation evaluates it once
and sums the integrated model bins onto each native RMF input grid. Detector
counts are compared per spectrum after folding with its own ARF, RMF, exposure,
and AREASCAL. The optional plans apply detector grouping independently in
joint mode. The prepared fit captures the selection, counts, and calibration
at the time you call `combine_datasets`. Rebuild it after changing those inputs.

Pass `stat` to `fit(problem, optimizer; stat)` with a SpectralFitting-compatible
optimizer. `grid_statistic` selects Cash deviance for raw source-only Poisson
counts, W-stat for measured Poisson backgrounds, and `ChiSquared()` for
supplied statistical errors on background-free data. The W-stat uses the
attached off counts and the source-to-background exposure, BACKSCAL, and
AREASCAL ratios for each member. Supply raw source and background spectra
for W-stat. If you already used SpectralFitting's `subtract_background!`,
set `data.spectrum.error_statistics = SpectralFitting.ErrorStatistics.Numeric`
before preparing a joint chi-squared fit so it uses the propagated errors.
For least squares, use SpectralFitting's `LevenbergMarquadt()`.

For example, with `OptimizationOptimJL` installed in your project:

```julia
using OptimizationOptimJL: NelderMead

result = fit(problem, NelderMead(); stat)
```

## Sum spectra on a reference detector grid

```julia
combined = combine_datasets(matched; mode=:sum, groupings=plans)
summed_problem = FittingProblem(source => combined)
summed_stat = grid_statistic(combined)
```

Sum mode takes background-free raw counts. It uses the reference member's
selected or grouped bins in wavelength (Å) by default. Counts and folded model
predictions are mapped from the original selected channels before reference
grouping, using bin-overlap fractions in wavelength. Use `coordinate=:energy`
for keV overlaps, or provide a custom output `grid=BinGrid(low, high)` in the
chosen detector coordinate. The output grid may crop coverage. Gaps and
separate bin boundaries are available on `combined.grid`.

Fractional overlap produces correlated counts. `combined.covariance` stores
the full sparse covariance, and chi-squared fitting whitens both data and
model with that covariance. A singular covariance requires a coarser
detector grid or a joint fit. For exact sums of Poisson counts,
`grid_statistic` selects Cash deviance. A fit using explicit statistical
errors uses their propagated variances.

For counts-space output, use `combined.counts` and
`predict_counts(combined, incident_flux)`, where `incident_flux` contains
bin-integrated model values on `combined.model_edges`. For example,
`incident_flux = invokemodel(combined.model_edges, source)` uses the current
model parameters. For a joint fit the same prediction function returns the
packed native-bin spectra, with `joint.member_ranges` giving the slice for
each instrument. The detector grids are available as `joint.members[i].grid`
in keV.

## Plot data, counts, fits, and unfolded flux

The [plotting guide](docs/plotting.md) includes runnable synthetic examples
for HETG overlays, RGS detector gaps, and multi-panel figures.
See [RGS channel scalings](docs/rgs-channel-scalings.md) for plotting and
fitting loaded RGS spectra with per-channel AREASCAL and BACKSCAL.

Add CairoMakie to the Julia environment where you use SpectralGrouping.

```julia
using SpectralFitting, SpectralGrouping, CairoMakie

rates = plot_data(heg; xaxis=:wavelength)
counts = plot_counts(matched; combine=:sum, xaxis=:wavelength, rebin=2)
fitted = plot_fit(result; xaxis=:wavelength, res=4)
unfolded = plot_unfold(heg; xaxis=:wavelength)
save("fit.pdf", fitted.figure)
```

`plot_data` defaults to count-rate density. Use `normalization=:rate` for
integrated rates or `:counts` for integrated counts. `plot_counts` uses
counts per detector bin. The default `combine=:none` shows observations
individually. To pool them, use `:sum` or `:mean`, or use `:exposure` for
total counts divided by total exposure. A `SummedDataset` uses its existing
target grid. Other pooled plots use the reference observation's original RMF
detector grid and gather eligible contributions from every observation.
Exposure normalization uses the total observation exposure across gaps.
Matching uses wavelength by default, with an energy fallback for a
zero-energy edge. Set `matching_coordinate=:energy` to use keV explicitly.
The display axis can be `:energy`, `:wavelength`, or `:frequency`.

Use `groupings` for individual native plots. For pooled plots, `rebin=N` groups
up to `N` consecutive common-grid bins after alignment. Use `detector_grid`,
`align_detector_grids`, and `rebin_detector_grid` to inspect the grid and
each observation's contributions. The
default `background=:subtract` shows on-region counts minus scaled off-region
counts. `:include` shows the on-region spectrum and `:only` shows the
scaled off-region background. The returned numerical curves preserve their
bin boundaries, validity, diagonal errors, and full propagated covariance.
Use `plot_curves(data; kind=:data)` to get those arrays without CairoMakie.

`plot_fit` uses fitted parameters. You can also pass `model=source` to
`plot_data`, `plot_counts`, or `plot_unfold`. For fit results with multiple
datasets, choose the fit slice using `index`. The residual panel accepts
these ISIS modes:

| Mode | Per spectrum | Per displayed group |
|---|---:|---:|
| Standardized residual or signed square root of deviance | 1 | 4 |
| Signed squared residual or signed deviance | 2 | 5 |
| Data/model ratio | 3 | 6 |

`residual_stat=:auto` uses the fitted statistic when supported. Use
`:chi2`, `:cash`, or `:wstat` to select a diagnostic explicitly. A W-stat
curve uses the off counts and source-to-background scale of each member.
Pooled fractional Cash and W-stat curves are display diagnostics. They do
not represent independent count likelihoods. Residuals use count-space
values before display normalization and scaling.

The result exposes `plot.axes.data` and `plot.axes.residual`. To create
side-by-side panels, construct the axes in the desired figure and pass
`axes=(data=top, residual=bottom)` and `figure=fig`. Each column's pair
shares its x limits, and the lower panel carries the x-axis labels.

`plot_unfold` integrates the ARF over wavelength in each RMF input bin,
then folds the result through the RMF and scales by PHA exposure and
AREASCAL. With no separate ARF it assumes a full response matrix that
already includes effective area. Counts and sensitivity are pooled before
division. The data-only path supports different ARF/RMF boundaries.
For an incident bin starting at 0 keV, unfolding uses a 10⁻¹⁰ keV
lower bound for the wavelength integral (or a smaller bound if needed to
fit within the response grids). A nonzero response in that bin makes its
sensitivity depend on this cutoff, which applies only to the plotted
wavelength integral.
A supplied model uses the fitting response; standalone model overlays
require aligned ARF/RMF input bins. For `plot_unfold(result)`, the folded
curve is the default. Use `model_display=:source` for the unsmeared source.
`flux_unit=:ergs` switches from photons to energy flux, and both
`plot_unfold` and `plot_counts` support ISIS-style `power` and
`power_scale`.

## Run the checks

```sh
julia --project=SpectralGrouping.jl -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```
