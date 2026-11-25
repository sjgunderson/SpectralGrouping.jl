module SpectralGroupingMakieExt

import Makie
import SpectralGrouping as SG

const SPECTRUM_COLORS = (:steelblue, :darkorange, :seagreen, :mediumpurple, :firebrick)

function _plotscale(scale)
    scale === :linear && return identity
    scale === :log && return log10
    return scale
end

_logscale(scale) = scale === log10 || scale === log || scale === log2

function _set_axis_scale!(ax, dim, scale)
    setting = dim == 1 ? ax.xscale : ax.yscale
    setting[] === scale && return nothing

    if _logscale(scale) && isempty(ax.scene.plots) && minimum(ax.targetlimits[])[dim] <= 0
        limits = Makie.convert_limit_attribute(ax.limits[])
        bounds = something(limits[dim], (nothing, nothing))
        positive = map(bound -> !isnothing(bound) && bound > 0 ? bound : nothing, bounds)
        default_low, default_high = Makie.defaultlimits(scale)
        low = something(positive[1], min(default_low, something(positive[2], default_high) / 10))
        high = something(positive[2], max(default_high, 10 * low))

        # Makie uses the old target limits while the axis has no plots.
        ax.limits[] = Base.setindex(limits, (low, high), dim)
        setting[] = scale
        ax.limits[] = Base.setindex(limits, all(isnothing, positive) ? nothing : positive, dim)
    else
        setting[] = scale
    end
    return nothing
end

function _visible(curve, xscale, yscale)
    good = curve.valid .& isfinite.(curve.y)
    _logscale(xscale) && (good .&= curve.grid.low .> 0)
    _logscale(yscale) && (good .&= curve.y .> 0)
    return good
end

function _curve_steps(curve, valid)
    x, y = Float64[], Float64[]
    grid = curve.grid
    previous = 0
    for i in eachindex(valid)
        valid[i] || continue
        if !iszero(previous) &&
                (previous + 1 != i || !SG._roundtrip_edge(grid.high[previous], grid.low[i]))
            push!(x, NaN)
            push!(y, NaN)
        end
        append!(x, (grid.low[i], grid.high[i]))
        append!(y, (curve.y[i], curve.y[i]))
        previous = i
    end
    return x, y
end

function _draw_curve!(ax, curve, color; width, marker, errors, model = false)
    xscale, yscale = ax.xscale[], ax.yscale[]
    valid = _visible(curve, xscale, yscale)
    artists = Any[]
    if any(valid)
        x, y = _curve_steps(curve, valid)
        push!(artists, Makie.lines!(ax, x, y; color, linewidth = width,
                                   linestyle = model ? :dash : :solid,
                                   label = model ? curve.label * " model" : curve.label))
        center = curve.grid.low ./ 2 .+ curve.grid.high ./ 2
        if marker > 0
            push!(artists, Makie.scatter!(ax, center[valid], curve.y[valid]; color, markersize = marker))
        end
        if errors && !isnothing(curve.error)
            keep = valid .& isfinite.(curve.error)
            _logscale(yscale) && (keep .&= curve.y .- curve.error .> 0)
            if any(keep)
                push!(artists, Makie.errorbars!(ax, center[keep], curve.y[keep], curve.error[keep]; color))
            end
        end
    end
    return artists
end

function _axes(figure, axes, has_residual, xscale, yscale)
    if isnothing(axes)
        fig = isnothing(figure) ? Makie.Figure() : figure
        top = Makie.Axis(fig[1, 1]; xscale = something(_plotscale(xscale), identity),
                         yscale = something(_plotscale(yscale), identity))
        lower = if has_residual
            ax = Makie.Axis(fig[2, 1]; xscale = top.xscale[])
            Makie.rowsize!(fig.layout, 2, Makie.Auto(false, 0.32))
            Makie.rowgap!(fig.layout, 4)
            ax
        end
        return fig, top, lower
    end
    top = axes.data
    lower = hasproperty(axes, :residual) ? axes.residual : nothing
    top isa Makie.Axis && (isnothing(lower) || lower isa Makie.Axis) ||
        throw(ArgumentError("axes must contain Makie Axis objects"))
    has_residual && isnothing(lower) && throw(ArgumentError("supply axes.residual when res is nonzero"))
    return figure, top, lower
end

function SG._render_plot(
    plot::SG.SpectrumPlot; figure, axes, xscale, yscale, color, linewidth,
    markersize, errorbars, legend,
)
    has_residual = !isempty(plot.residuals)
    fig, top, lower = _axes(figure, axes, has_residual, xscale, yscale)
    if !isnothing(axes) && !isnothing(xscale)
        _set_axis_scale!(top, 1, _plotscale(xscale))
    end
    if !isnothing(axes) && !isnothing(yscale)
        _set_axis_scale!(top, 2, _plotscale(yscale))
    end
    top.xlabel = plot.xlabel
    top.ylabel = plot.ylabel
    if has_residual
        _set_axis_scale!(lower, 1, top.xscale[])
        lower.xlabel = plot.xlabel
        lower.ylabel = plot.residual_label
        Makie.linkxaxes!(top, lower)
        Makie.hidexdecorations!(top; ticks = false, grid = false, minorgrid = false, minorticks = false)
        if isempty(lower.scene.plots)
            Makie.hlines!(lower, [plot.residual_baseline]; color = :gray, linestyle = :dash)
        end
    end

    observation_colors = [isnothing(color) ? SPECTRUM_COLORS[mod1(i, length(SPECTRUM_COLORS))] : color
                          for i in eachindex(plot.data)]
    data_artists = [_draw_curve!(top, curve, observation_colors[i]; width = linewidth,
                                 marker = markersize, errors = errorbars)
                    for (i, curve) in enumerate(plot.data)]
    model_artists = [_draw_curve!(top, curve, observation_colors[i]; width = linewidth,
                                  marker = 0, errors = false, model = true)
                     for (i, curve) in enumerate(plot.model)]
    residual_colors = length(plot.residuals) <= length(observation_colors) ? observation_colors :
        [isnothing(color) ? SPECTRUM_COLORS[mod1(i, length(SPECTRUM_COLORS))] : color
         for i in eachindex(plot.residuals)]
    residual_artists = [_draw_curve!(lower, curve, residual_colors[i];
                                     width = linewidth, marker = markersize, errors = errorbars)
                        for (i, curve) in enumerate(plot.residuals)]
    legend && any(artists -> !isempty(artists), data_artists) && Makie.axislegend(top)
    return (; figure = fig, axes = (; data = top, residual = lower), curves = plot,
            artists = (; data = data_artists, model = model_artists, residual = residual_artists))
end

end
