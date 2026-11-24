# Plotting spectra and fit results

The [synthetic example](examples/plotting.jl) runs without FITS files and saves
a two-column HETG figure and an RGS gap figure. Add AstroRegistry as described
in the [installation instructions](../README.md#install), then run these
commands from the repository root to use a separate Julia environment.

```sh
julia --project=/tmp/spectral-plot-demo -e 'using Pkg; Pkg.develop(path="."); Pkg.add(["SpectralFitting", "CairoMakie", "ForwardDiff", "Optimization", "OptimizationOptimJL"])'
julia --project=/tmp/spectral-plot-demo docs/examples/plotting.jl /tmp/spectral-plots
```

When working in a notebook, import the packages below. Replace the example
observations with your loaded SpectralFitting data and use your own source
model for `plasma`. The synthetic script uses a power law. The
[multi-grid W-stat guide](multi-grid-wstat.md) builds the two-slice `result`
used in the HETG examples.

```julia
using SpectralGrouping, CairoMakie
```

## One observation

```julia
counts = plot_counts(megm1; xaxis = :energy, model = plasma)
density = plot_data(megm1; xaxis = :wavelength)
rate = plot_data(megm1; xaxis = :frequency, normalization = :rate, yscale = :log)
flux = plot_unfold(megm1; xaxis = :wavelength, flux_unit = :photons)
energy_flux = plot_unfold(megm1; xaxis = :energy, flux_unit = :ergs, xscale = :log)
```

`plot_counts` displays counts per detector bin. `plot_data` defaults to count
rate density per keV, Å, or Hz, depending on `xaxis`. Use `normalization=:rate`
for an integrated count rate or `:counts` for integrated counts. The
rendering keywords `xscale` and `yscale` accept `:linear` or `:log`.

`plot_unfold` divides by detector sensitivity derived from the response and
exposure and converts the result to flux density in the displayed coordinate.
It can run on data alone, including an ARF with a different incident grid
from the RMF. For pooled spectra, counts and sensitivity are projected and
summed before division. The curve is blank wherever pooled sensitivity is
zero. For energy-flux plots in energy or frequency, the default power
weighting produces `E F_E` or `ν F_ν`. Use `power=1` in wavelength or
`power=2` in energy and frequency for energy flux density.

## Current and fitted source parameters

```julia
current = plot_data(megm1; model = plasma)
fitted_meg = plot_fit(result; index = 1, combine = :sum)
fitted_heg = plot_fit(result[2]; combine = :sum)
folded_flux = plot_unfold(result; index = 1, combine = :sum)
source_flux = plot_unfold(result[1]; combine = :sum, model_display = :source)
```

`model=plasma` evaluates the object's current parameters. A fit result supplies
the fitted parameters for its selected model/data pair. Use `index` on the
full result or pass a slice directly. Count and rate overlays fold the source
through the response. The default unfolded overlay corrects the folded model
by the same sensitivity as the data. `model_display=:source` shows the
incident source on the model grid within covered detector spans. Both are
display conventions.

## Coadding RGS through detector gaps

```julia
rgs = match_dataset_grids(rgs1, rgs2; reference = 2)
before = plot_counts(rgs; combine = :sum, xaxis = :wavelength, rebin = 1)
after = plot_counts(rgs; combine = :sum, xaxis = :wavelength, rebin = 4)
unfolded = plot_unfold(rgs; combine = :sum, xaxis = :wavelength, rebin = 4)
rgs1_target = plot_counts(rgs; combine = :sum, reference = 1, xaxis = :wavelength)
individual = plot_counts((rgs1, rgs2); combine = :none, xaxis = :wavelength)
```

For a pooled raw collection, `GridMatch`, or `JointDataset` (including a fit
result slice), the chosen reference supplies its original RMF detector
channel boundaries by default. A raw collection defaults to observation 1.
A match or joint dataset reuses its resolved detector target. `reference`
on the plot selects that observation's original grid. A prepared
`SummedDataset` uses its fitted grid.

Only good-quality, noticed channels contribute. If an RGS1 channel is bad
and RGS2 covers the same interval, the RGS2 measurement survives on the
RGS1 grid. An interval missing from the reference EBOUNDS has no target bin.
Intervals with target bins and no contributing observation are blank.
Covered zero-count bins remain valid. The overlap integration crops at the
chosen target's boundaries. Offset references can give different bins and
totals if their extents differ. Wavelength is the default overlap
coordinate. Set `matching_coordinate=:energy` for overlaps in keV
independently of the displayed `xaxis`. If any input to the overlap has a
zero-energy edge, the default overlap coordinate switches to energy.
A zero-energy target edge cannot be displayed in wavelength. For an
unfolded energy or frequency plot, a bin starting at zero stays blank
because its wavelength span is infinite.

Replace a reference `:width` plan passed through `groupings` on a pooled plot
with `rebin=N`. Each observation is projected separately onto the full
ungrouped target before the same width grouping is applied to all of them.
Groups start at the lower edge of each contiguous segment in the matching
coordinate. A segment boundary or final remainder can leave a short group.
Masks leave the grouping phase intact. A group with zero coverage across all
members stays blank. Partially covered groups contain the measured counts
and sensitivity, with counts and sensitivity summed before unfolding.
A uniform 0.010 Å target gives 0.040 Å full-width groups for `rebin=4`.

Use native `groupings` for individual plots or when preparing a joint fit.
A pooled plot from a joint dataset or fit that used the same `:width` target
for every observation defaults to that width on the shared display grid.
For example, a joint fit built with 20-channel width plans shows one error
bar at the center of each displayed group. Pass `rebin` explicitly to
override the width, or to select a shared width when the fit plans differ
or use count or S/N targets. `rebin=1` selects the ungrouped display target.
A plot from a grouped fit projects the fitted bins that are available.
This projection assumes constant density within each fitted bin and cannot
recover its original measurements. `plot_curves` exposes bin edges,
validity, errors, and covariance without a graphics backend.

```julia
target = detector_grid(rgs; coordinate = :wavelength)
alignment = align_detector_grids(rgs; unfold = true)
grouped = rebin_detector_grid(alignment, 4)
widths_in_angstrom = grouped.grid.high .- grouped.grid.low
grouped.channels_per_bin
grouped.members[1].coverage  # geometric fraction measured by RGS1
grouped.members[2].weights   # folded response sensitivity from RGS2
```

Each member retains its own counts, covariance, scaled-background inputs,
and folded prediction when a model or fit result is supplied. Coverage is a
geometric fraction in the matching coordinate. It is separate from response
sensitivity and exposure. Quantitative comparison with SAS also depends on
the per-channel AREASCAL and BACKSCAL calibration.

Supply a complete detector grid explicitly if the reference EBOUNDS lacks
the edges you need or if you want uniform widths in a particular coordinate.
`detector_grid` takes a paired `BinGrid` or a vector of contiguous edges.
`grid=` continues to set the incident model grid in keV.

```julia
uniform_rgs = match_dataset_grids(rgs1, rgs2;
    detector_grid = 10.0:0.010:24.0, detector_coordinate = :wavelength)
plot_unfold(uniform_rgs; combine = :sum, xaxis = :wavelength, rebin = 4)
```

`background=:subtract` displays source-region counts minus each member's
scaled off counts. `:include` shows the source-region counts and adds the
scaled off counts to any model overlay. `:only` shows scaled off counts.
Display coadding supports attached backgrounds. The fitting operation
`combine_datasets(...; mode=:sum)` has separate restrictions and requires
background-free raw counts. Use joint mode for native on/off W-stat.

For rates, `combine=:sum` adds individual rates, and `:mean` averages them.
`:exposure` divides total counts by total observation exposure,
and `:mean_exposure` multiplies that rate by the number of observations.
These use global coefficients through gaps. For a rate averaged by the
effective exposure in each bin, account for detector coverage and exposure
per bin separately.

The reference-grid choice follows
[ISIS grid matching](https://github.com/houckj/isis/blob/bc8e15a9aeecace0679fb9a90c6e8cabc551cfe6/share/hist-cmds.sl#L3034-L3058)
and its [overlap rebinning](https://github.com/houckj/isis/blob/bc8e15a9aeecace0679fb9a90c6e8cabc551cfe6/src/util.c#L1189-L1242).
For the RGS quality rule, see the [SAS rgscombine algorithm](https://xmm-tools.cosmos.esa.int/external/sas/current/doc/rgscombine/node8.html).

## Shared axes

```julia
pmeg = plot_fit(result; index = 1, xaxis = :wavelength, combine = :sum,
                res = 4, color = :steelblue, linewidth = 2)
pheg = plot_fit(result; index = 2, xaxis = :wavelength, combine = :sum,
                res = 4, color = :darkorange, markersize = 4, errorbars = false,
                axes = pmeg.axes, figure = pmeg.figure)
axislegend(pmeg.axes.data)
```

Pass `axes=pmeg.axes` to reuse both panels. Passing only `figure` creates new
axes. `axes=(data=ax,)` suffices with `res=0`. With residuals enabled, supply
both `data` and `residual` axes. For clarity use the same coordinate, units,
and residual mode on an overlay.

## Separate panels

```julia
fig = Figure(size = (1000, 500))
left = (data = Axis(fig[1, 1]; title = "MEG"), residual = Axis(fig[2, 1]))
right = (data = Axis(fig[1, 2]; title = "HEG"), residual = Axis(fig[2, 2]))
plot_fit(result; index = 1, axes = left, figure = fig, combine = :sum, res = 4)
plot_fit(result; index = 2, axes = right, figure = fig, combine = :sum, res = 4)
rowsize!(fig.layout, 2, Auto(false, 0.32))
rowgap!(fig.layout, 6)
xlims!(left.residual, 1.0, 4.7)
xlims!(right.residual, 1.7, 6.4)
save("hetg-panels.pdf", fig)
```

The plotting call links the two x-axes within each pair and puts x tick
labels on the residual axis. Each pair can use its own limits.

```julia
stacked = Figure(size = (720, 700))
ccd_layout = GridLayout(stacked[1, 1])
rgs_layout = GridLayout(stacked[2, 1])
ccd_axes = (data = Axis(ccd_layout[1, 1]; title = "PN and MOS"),
            residual = Axis(ccd_layout[2, 1]))
rgs_axes = (data = Axis(rgs_layout[1, 1]; title = "RGS"),
            residual = Axis(rgs_layout[2, 1]))
plot_data(pn; model = plasma, axes = ccd_axes, figure = stacked, res = 1, color = :steelblue)
plot_data(mos; model = plasma, axes = ccd_axes, figure = stacked, res = 1,
          color = :darkorange, legend = true)
plot_data(rgs; model = plasma, axes = rgs_axes, figure = stacked, combine = :sum,
          xaxis = :wavelength, res = 4)
for layout in (ccd_layout, rgs_layout)
    rowsize!(layout, 2, Auto(false, 0.32))
    rowgap!(layout, 6)
end
xlims!(ccd_axes.residual, 0.5, 7)
xlims!(rgs_axes.residual, 11, 25)
```

## Residuals

| Display | Per spectrum | Per displayed group |
| --- | ---: | ---: |
| Hide residuals | 0 | 0 |
| Standardized χ² residual or signed square root of Cash or W deviance | 1 | 4 |
| Signed squared χ² residual or signed Cash or W deviance | 2 | 5 |
| Data/model ratio | 3 | 6 |

`residual_stat=:auto` recognizes the fitted statistic, including joint
W-stat, and defaults to χ² for a current model. Use `:chi2`, `:cash`, or
`:wstat` for an explicit diagnostic. W-stat uses the measured off counts and
each member's scale. Residuals are computed in counts before display
normalization. Coadded or rebinned residuals are display diagnostics.
Fractional projections in particular do not define independent Poisson
measurements. To inspect per-order W-stat residuals on the selected
likelihood bins, use the unpooled, unreduced detector grid.

```julia
per_order = plot_fit(result; index = 1, combine = :none, res = 1, rebin = 1,
                     residual_stat = :auto)
```

## Styling and export

Use `color`, `linewidth`, `markersize`, `errorbars`, and `legend` on each
rendering call. The returned `artists.data`, `artists.model`, and
`artists.residual` hold separate collections of editable CairoMakie
plots. Each collection contains one vector of artists per curve.

```julia
for collection in (pmeg.artists.model, pheg.artists.model), group in collection, line in group
    line.color = :black
end
pmeg.axes.data.title = "MEG and HEG"
pmeg.axes.residual.xticks = 2:2:12
xlims!(pmeg.axes.residual, 2, 12)
save("hetg-overlay.png", pmeg.figure)
```
