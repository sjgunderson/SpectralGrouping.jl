# RGS channel scalings

Use `with_channel_scalings` on each loaded, ungrouped `SpectralFitting.SpectralData`.
Provide its original source-region PHA and the measured background PHA already
associated with it. Check `HDUCLAS2` when choosing XMM-Newton SAS products.
Use the `TOTAL` source spectrum and the `BKG` background spectrum. For
example, the public TW Hya sample uses `SBSPEC` and `BGSPEC` for these
roles. Its `SRSPEC` is marked `NET` and already has the background subtracted.

In the Julia project used by the notebook, load the checkout containing this
change with `import Pkg; Pkg.develop(path="/path/to/SpectralGrouping.jl")` and
restart the kernel.

Keep `rgs_source_phas` and `rgs_background_phas` in the same order as `rgslist`.
Apply energy masks either before or after attaching the channel scalings.

```julia
import SpectralGrouping as SG
import SpectralFitting as SF

@assert length(rgslist) == length(rgs_source_phas) == length(rgs_background_phas)
rgs_scaled = [
    SG.with_channel_scalings(d; source=s, background=b)
    for (d, s, b) in zip(rgslist, rgs_source_phas, rgs_background_phas)
]
for d in rgs_scaled
    SF.mask_energies!(d, 0.25, 3)
end
matchedrgs = SG.match_dataset_grids(rgs_scaled)
jointrgs = SG.combine_datasets(matchedrgs; mode=:joint)
```

`SF.mask_energies!` narrows the existing mask. To select a new energy range
from all channels, call `fill!(d.data_mask, true)` before masking again.
SpectralGrouping still excludes bad-quality source channels. To update a
prepared fit after changing the mask, regenerate any grouping plans, run
`combine_datasets` again on the existing match, and rebuild the fit problem,
statistic, and configuration.

The helper reads the `SPECTRUM` table and matches source and background
rows by `CHANNEL`. For each scale it uses the column, then the scalar
keyword, then one. It leaves the raw counts in place. For channel `i`,
the model response includes `source_exposure * source_AREASCAL[i]` on the
detector row. The off-to-on background factor is

```text
alpha[i] = (source_exposure * source_AREASCAL[i] * source_BACKSCAL[i]) /
           (background_exposure * background_AREASCAL[i] * background_BACKSCAL[i])
```

Masked source channels can carry zero scale values, as in RGS chip gaps.
An active channel with invalid scales or bad background quality raises an
error when preparing the fit or a background-subtracted plot. Mask the
affected source channel and rebuild the joint dataset.

To inspect the first few channels in the prepared data:

```julia
m = first(jointrgs.members)
i = 1:min(5, length(m.counts))
(; energy_low=m.grid.low[i], on=m.counts[i], off=m.background[i],
   alpha=m.alpha[i], net=m.counts[i] .- m.alpha[i] .* m.background[i])
```

Compare the net, on-region, and scaled off-region spectra on the same
display grid. `combine=:exposure` normalizes pooled counts by total nominal
exposure, and `rebin=5` groups the displayed bins. The same `rebin=5`
keyword works when plotting a single scaled spectrum.

```julia
using CairoMakie

net = SG.plot_data(jointrgs; combine=:exposure, background=:subtract, rebin=5)
on  = SG.plot_data(jointrgs; combine=:exposure, background=:include, rebin=5)
off = SG.plot_data(jointrgs; combine=:exposure, background=:only, rebin=5)
```

## Grouping for the fit

Build the plans from the scaled spectra after setting the masks. Here
`:width` creates groups of up to five contiguous channels. Use `:counts`
for a minimum observed count target or `:ogip` to use the file's grouping
flags.

```julia
plansrgs = [SG.grouping_plan(d, :width, 5) for d in rgs_scaled]
jointrgs = SG.combine_datasets(matchedrgs; mode=:joint, groupings=plansrgs)
```

`SG.plot_data(jointrgs; combine=:exposure, xaxis=:wavelength)` projects the
selected channels onto the shared detector lattice and groups it by five.
This includes target bins in intervals masked by the reference. Use
`rebin=N` to choose a different display width.

For an individual spectrum, `SG.group_width(rgs1spec_scaled, 5)` returns
a fresh calibrated spectrum with the grouping applied. It can be plotted
directly or passed to `match_dataset_grids` for fitting.

Each fit bin contains the summed raw on counts, summed raw off counts, and
the sum of the folded model after applying each channel's response scale.
For grouped W-stat, the background factor follows [Sherpa's W-stat
convention](https://sherpa.readthedocs.io/en/4.19.0/_modules/sherpa/stats.html#WStat.calc_stat).
Within each group, replace each of the four scale columns in the expression
for `alpha` by the midpoint of its minimum and maximum. Plots of the net
spectrum still sum the individually scaled off counts and propagate their
variances. The plotted net counts can therefore differ from the grouped
W-stat's raw on counts minus its group factor times the raw off counts.
Omit `groupings` to use one W-stat contribution per selected native channel.

For a PN, MOS, and RGS fit, create a new fitting problem with the prepared
`jointrgs`, reapply parameter bindings, and create a fresh config using
`stat=SG.grid_statistic(jointpn, jointmos, jointrgs)`.

Use `SG.plot_fit(result; index=3, combine=:none, rebin=1, res=1,
residual_stat=:wstat)` for residuals on the RGS fit bins if RGS is the third
fit slice. For additional display rebinning that mixes different fit
`alpha` values or splits fit bins, use `res=0` or `residual_stat=:chi2`.
