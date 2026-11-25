# Joint W-stat on multiple incident grids

Prepare each group of detector spectra on the incident grid appropriate to
its response. For Chandra HETG, put the two MEG orders in one group and the two
HEG orders in another. Each source model is evaluated on its group's incident
grid and folded independently through each order's response. The fit sums four
native on/off W-stat contributions into two dataset subtotals.

```julia
import SpectralFitting as SF
import SpectralGrouping as SG

# The four loaded orders have attached, channel-aligned backgrounds.
spectral_sets = [(megm1, megp1), (hegm1, hegp1)]
joints = map(spectral_sets) do spectra
    SG.combine_datasets(SG.match_dataset_grids(spectra...); mode = :joint)
end

# One source model per incident grid, in the same order as joints.
models = [meg_source, heg_source]
problem = SF.FittingProblem((models[i] => joints[i] for i in eachindex(joints))...)

# Share every corresponding free parameter across model slots.
SG.tie_all_free!(problem)
stat = SG.grid_statistic(joints...)
cfg = SF.FittingConfig(problem; stat)

total_wstat(p) = sum(i -> SF.measure_objective!(cfg, p, i), eachindex(joints))
total_wstat(SF.get_value.(cfg.u0))
```

The same pattern works for any number of prepared joint datasets and for
groups with different numbers of orders. Each dataset must select W-stat by
itself, which requires raw integer counts and measured background. Use the
existing single-argument form when there is one joint dataset. Each order
retains its source counts, off-region counts, and source-to-background
scaling. Coadd orders for display after fitting.

Reusing a Julia model object creates separate SpectralFitting model entries
with independent free parameters. `SG.tie_all_free!(problem)` ties the
corresponding free parameters to model slot 1. It resolves component
labels for composites, including wrapped composites, and checks that every
slot has the same free parameter selectors before making any ties. Call it
before creating `SF.FittingConfig`. For a subset of parameters, use
`SF.bindall!` or `SF.bind!` so parameters such as cross-calibration constants
can vary independently. For repeated names in a composite model, qualify
the component, for example
`SF.bindall!(problem, (:a1, :K))` or
`SF.bind!(problem, (1, :a1, :K) => (2, :a1, :K))`. For cached Plasma.jl models,
build one model per incident grid with the same AtomDB object. This gives each
model a separate synthesis cache while sharing the atomic tables.

## Bounded Optimization.jl fit

Install Optimization, OptimizationOptimJL, and ForwardDiff in the notebook
environment. Paste the [bounded fit helper](examples/bounded-wstat.jl) into a
cell or include it from the repository checkout.

```julia
include("docs/examples/bounded-wstat.jl")
result = fit_bounded_wstat(cfg)

meg_wstat, heg_wstat = result.stats   # HETG example
global_wstat = sum(result.stats)
```

The helper reads the initial free parameters and bounds from `cfg.u0`. Each
initial value must lie strictly inside its bounds. The solver uses
`Optim.Fminbox(Optim.LBFGS())` and ForwardDiff to optimize scaled coordinates
`p = x .* scale`. `SF.finalize_result` receives the parameters in physical
units. The result's `u` holds those physical parameters, while the embedded
optimizer solution's `u` holds the scaled coordinates.

`SF.measure_objective!(cfg, p, i)` evaluates one model/data pair. The general
`SF.fit` optimizer path sums all pairs internally, and the bounded helper
sums them explicitly. Use `result.stats[i]` for one pair and
`sum(result.stats)` for the global W-stat. In SpectralFitting 0.7.x,
`SF.measure(stat, result)` reports the first result slice.
