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
prepared fit after changing the mask, run `combine_datasets` again on the
existing match and rebuild the fit problem, statistic, and configuration.

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
keyword works when plotting a single scaled spectrum. The fit uses native
channels to retain each on/off count pair and its background scale.

```julia
using CairoMakie

net = SG.plot_data(jointrgs; combine=:exposure, background=:subtract, rebin=5)
on  = SG.plot_data(jointrgs; combine=:exposure, background=:include, rebin=5)
off = SG.plot_data(jointrgs; combine=:exposure, background=:only, rebin=5)
```

For a PN, MOS, and RGS fit, create a new fitting problem with this `jointrgs`,
reapply parameter bindings, and create a fresh config using
`stat=SG.grid_statistic(jointpn, jointmos, jointrgs)`. Fit the calibrated
RGS on its native channels by omitting `groupings` when preparing `jointrgs`.
The W-stat uses each channel's raw on and off counts and its own `alpha`.

Use `SG.plot_fit(result; index=3, combine=:none, rebin=1, res=1,
residual_stat=:wstat)` for native RGS W-stat residuals if RGS is the third
fit slice. Bins that pool different `alpha` values within a spectrum or
split native counts across bins cannot use the W-stat residual display.
For those displays, use `res=0` or `residual_stat=:chi2`.
