function _plot_statistic(requested, fitted)
    if requested == :auto
        fitted isa _JointWStat && return :wstat
        if fitted isa _GridPoisson
            return isnothing(fitted.background) ? :cash : :wstat
        end
        fitted isa SF.Cash && return :cash
        return :chi2
    end
    requested in (:chi2, :cash, :wstat) ||
        throw(ArgumentError("residual_stat must be :auto, :chi2, :cash, or :wstat"))
    return requested
end

function _pooled_profile(on, source, offsets, scales)
    # Equal scales have the same nuisance parameter after pooling the off data.
    pooled = Dict{Float64,Float64}()
    for (off, alpha) in zip(offsets, scales)
        pooled[alpha] = get(pooled, alpha, 0.0) + off
    end
    if length(pooled) == 1
        scale, off = only(pooled)
        return _profile_deviance(on, source, off, scale)
    end
    largest = maximum(keys(pooled))
    lower = -inv(largest)
    function on_mean(t)
        return source + sum((alpha * off / (1 + alpha * t)
                             for (alpha, off) in pooled if off > 0); init = 0.0)
    end
    function equation(t)
        return (1 - t) * on_mean(t) - on
    end
    if iszero(on)
        t = 1.0
    elseif iszero(pooled[largest]) && equation(lower) <= 0
        # A zero-count off region can absorb the remaining expected background.
        mean_on = on / (1 - lower)
        contribution = _poisson_deviance(on, mean_on)
        for (alpha, off) in pooled
            mean_off = if alpha == largest
                max(0.0, (mean_on - on_mean(lower)) / alpha)
            elseif iszero(off)
                0.0
            else
                off / (1 + alpha * lower)
            end
            contribution += _poisson_deviance(off, mean_off)
        end
        return contribution
    else
        lo, hi = lower, 1.0
        for _ in 1:80
            mid = (lo + hi) / 2
            (mid == lo || mid == hi) && break
            if equation(mid) > 0
                lo = mid
            else
                hi = mid
            end
        end
        t = hi
    end
    return _poisson_deviance(on, on_mean(t)) +
        sum((_poisson_deviance(off, off / (1 + alpha * t))
             for (alpha, off) in pooled if off > 0); init = 0.0)
end

function _append_off_regions!(counts, scales, background::_PlotBackground, bin)
    background.alpha > 0 || return nothing
    off = background.counts[bin]
    off >= 0 || throw(ArgumentError("W-stat requires raw nonnegative off-region counts"))
    push!(counts, off)
    push!(scales, background.alpha)
    return nothing
end

function _append_off_regions!(counts, scales, background::_ChannelPlotBackground, bin)
    # Fractional mappings are handled by _projected_wstat before pooling.
    overlap = background.fit_overlap
    rows = rowvals(overlap)
    for entry in nzrange(overlap, bin)
        row = rows[entry]
        off = background.off_counts[row]
        off >= 0 || throw(ArgumentError("W-stat requires raw nonnegative off-region counts"))
        push!(counts, off)
        push!(scales, background.alpha[row])
    end
    return nothing
end

function _fit_bin_wstat(member)
    on, source, background = member.counts, member.model, member.background
    any(<(0), on) && throw(ArgumentError("Poisson residuals require raw nonnegative on-region counts"))
    if isnothing(background) || (background isa _PlotBackground && iszero(background.alpha))
        return _poisson_deviance.(on, source), on .- source
    end
    off = background isa _ChannelPlotBackground ? background.off_counts : background.counts
    any(<(0), off) && throw(ArgumentError("W-stat requires raw nonnegative off-region counts"))
    alpha = background.alpha
    return _profile_deviance.(on, source, off, alpha), on .- source .- alpha .* off
end

function _projected_wstat(fit_members, target, covered, signed_squared)
    deviance = zeros(length(target.grid))
    difference = zeros(length(target.grid))
    for member in fit_members
        score, excess = _fit_bin_wstat(member)
        grid, order = _convert_plot_grid(member.grid, member.coordinate, target.coordinate)
        projection = overlap_projection(grid, target.grid)
        # Project the fitted contributions before applying the sign or square root.
        deviance .+= projection * score[order]
        difference .+= projection * excess[order]
    end
    values = fill(NaN, length(deviance))
    for i in eachindex(values)
        covered[i] && isfinite(deviance[i]) || continue
        magnitude = max(0.0, deviance[i])
        values[i] = sign(difference[i]) * (signed_squared ? magnitude : sqrt(magnitude))
    end
    return values
end

function _residual_curve(members, fit_members, xaxis, mode, statistic, background)
    first_member = first(members)
    grid, order = _convert_plot_grid(first_member.grid, first_member.coordinate, xaxis)
    n = length(grid)
    covered = _display_coverage(members, :include)
    ratio = mod1(mode, 3) == 3
    signed_squared = mod1(mode, 3) == 2
    fractional = any(m -> m.fractional, members)
    if !ratio && statistic == :wstat && fractional &&
            any(m -> m.background isa _ChannelPlotBackground, members)
        values = _projected_wstat(fit_members, first_member, covered, signed_squared)[order]
        return PlotCurve(grid, values, nothing, nothing, BitVector(isfinite.(values)),
                         _plot_label(members), fractional, nothing)
    end

    observed = reduce(+, (m.counts for m in members))
    source = reduce(+, (m.model for m in members))
    variance = reduce(+, (diag(m.covariance) for m in members))
    estimate = zeros(n)
    for member in members
        off = member.background
        if !isnothing(off)
            scaled, covariance = _scaled_background(off)
            estimate .+= scaled
            variance .+= diag(covariance)
        end
    end
    if !ratio && statistic in (:cash, :wstat) && any(<(0), observed)
        throw(ArgumentError("Poisson residuals require raw nonnegative on-region counts"))
    end

    values = fill(NaN, n)
    error = ratio || (statistic == :chi2 && !signed_squared) ? fill(NaN, n) : nothing
    off_counts, scales = Float64[], Float64[]
    for i in eachindex(observed)
        covered[i] || continue
        on, signal, off = observed[i], source[i], estimate[i]
        isfinite(signal) || continue
        if ratio
            numerator = background == :include ? on : on - off
            denominator = background == :include ? signal + off : signal
            denominator > 0 || continue
            values[i] = numerator / denominator
            error[i] = sqrt(variance[i]) / denominator
        elseif statistic == :chi2
            variance[i] > 0 || continue
            residual = (on - off - signal) / sqrt(variance[i])
            values[i] = signed_squared ? sign(residual) * residual^2 : residual
            isnothing(error) || (error[i] = 1.0)
        else
            signal >= 0 || continue
            empty!(off_counts)
            empty!(scales)
            if statistic == :wstat
                for member in members
                    bkg = member.background
                    if !isnothing(bkg) && member.coverage[i] > 0
                        _append_off_regions!(off_counts, scales, bkg, i)
                    end
                end
            end
            deviance = isempty(off_counts) ? _poisson_deviance(on, signal + off) :
                _pooled_profile(on, signal, off_counts, scales)
            if isfinite(deviance)
                estimated_off = isempty(off_counts) ? off : dot(scales, off_counts)
                signed = sign(on - signal - estimated_off)
                values[i] = signed * (signed_squared ? deviance : sqrt(max(0.0, deviance)))
            end
        end
    end
    ordered = values[order]
    return PlotCurve(grid, ordered, isnothing(error) ? nothing : error[order], nothing,
                     BitVector(isfinite.(ordered)), _plot_label(members), fractional, nothing)
end
