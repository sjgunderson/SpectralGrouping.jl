using CairoMakie: Axis, Figure, save

@testset "Independently editable CairoMakie panels" begin
    data = dataset(spectrum(1:3, [3, 4, 5]); low = [1.0, 2, 4])
    source = GridTestSource(SF.FitParam(0.1))
    figure = Figure()
    left = (data = Axis(figure[1, 1]), residual = Axis(figure[2, 1]))
    right = (data = Axis(figure[1, 2]), residual = Axis(figure[2, 2]))
    a = plot_counts(data; model = source, res = 1, axes = left, figure)
    b = plot_counts(data; model = source, res = 3, axes = right, figure)
    @test a.figure === b.figure === figure
    @test a.axes.data === left.data
    @test b.axes.residual === right.residual
    @test !left.data.xlabelvisible[] && !left.data.xticklabelsvisible[]
    @test left.residual.xlabelvisible[]
    b.axes.residual.ylabel = ""
    @test a.axes.residual.ylabel[] == "Residual (σ)"
    extension = Base.get_extension(SpectralGrouping, :SpectralGroupingMakieExt)
    curve = only(a.curves.data)
    steps, _ = extension._curve_steps(curve, curve.valid)
    @test count(isnan, steps) == 1
    automatic = plot_counts(data; model = source, res = 1, xscale = :log)
    @test automatic.axes.data.xscale[] === log10
    @test !automatic.axes.data.xlabelvisible[]
    mktempdir() do folder
        path = joinpath(folder, "panels.svg")
        save(path, figure)
        @test filesize(path) > 0
    end
end

@testset "Log scales on axes supplied by the caller" begin
    pn = dataset(spectrum(1:5, [11, 2, 3, 18, 30]); low = collect(0.0:4.0),
                 background = spectrum(1:5, [1, 2, 4, 3, 2]))
    mos = dataset(spectrum(1:3, [8, 13, 18]))
    fig = Figure()
    top = Axis(fig[1, 1]; limits = (nothing, (0.0, 10.0)))
    residual = Axis(fig[2, 1])
    axes = (; data = top, residual)
    source = GridTestSource(SF.FitParam(0.1))
    p = plot_data(pn; figure = fig, axes, model = source, res = 1, xscale = :log, yscale = :log)
    @test p.axes.data === top
    @test top.xscale[] === top.yscale[] === residual.xscale[] === log10
    @test residual.yscale[] === identity
    @test all(>(0), minimum(top.targetlimits[]))
    @test top.limits[] == (nothing, (nothing, 10.0))
    @test length(last(only(p.artists.data)).val_low_high[]) == 2
    overlay = plot_data(mos; figure = fig, axes = p.axes, xscale = :log, yscale = :log)
    @test overlay.axes.data === top

    blank_fig = Figure()
    blank_axis = Axis(blank_fig[1, 1])
    blank = plot_counts(dataset(spectrum(1:2, [0, -1])); axes = (data = blank_axis,),
                        xscale = :log, yscale = :log)
    @test isempty(only(blank.artists.data))
    @test blank_axis.limits[] == (nothing, nothing)
    @test all(>(0), minimum(blank_axis.targetlimits[]))
end

@testset "CairoMakie shared-width RGS plot" begin
    rgs1, rgs2, _ = rgs_width_pair()
    matched = match_dataset_grids(rgs1, rgs2)
    display = plot_counts(matched; combine = :sum, xaxis = :wavelength,
                          background = :include, rebin = 4)
    curve = only(display.curves.data)
    @test isapprox(curve.grid.high .- curve.grid.low, fill(0.040, 3))
    @test isapprox(curve.y, [4, 8, 12])
    mktempdir() do folder
        path = joinpath(folder, "rgs-shared-width.png")
        save(path, display.figure)
        @test filesize(path) > 0
    end
end

@testset "One error bar per pooled bin from a width-grouped joint fit" begin
    rgs1, rgs2, wave = rgs_width_pair(; n = 60, good1 = [1:20; 41:60], good2 = 21:60)
    matched = match_dataset_grids(rgs1, rgs2)
    plans = [grouping_plan(d, :width, 20) for d in (rgs1, rgs2)]
    joint = combine_datasets(matched; mode = :joint, groupings = plans)
    opts = (; combine = :sum, xaxis = :wavelength, background = :include)

    plotted = plot_data(joint; opts...)
    curve = only(plotted.curves.data)
    @test isapprox(curve.grid.low, wave[[1, 21, 41]])
    @test isapprox(curve.grid.high, wave[[21, 41, 61]])
    bars = last(only(plotted.artists.data)).val_low_high[]
    @test length(bars) == 3
    @test isapprox([bar[1] for bar in bars], (wave[[1, 21, 41]] .+ wave[[21, 41, 61]]) ./ 2)
end

include(joinpath(@__DIR__, "..", "docs", "examples", "plotting.jl"))

@testset "Runnable plotting guide" begin
    mktempdir() do folder
        demo = PlottingExample.demo(; output_dir = folder)
        @test demo.pmeg.axes.data === demo.pheg.axes.data
        @test demo.pmeg.axes.residual === demo.pheg.axes.residual
        @test demo.pmeg.figure === demo.pheg.figure
        @test demo.pmeg.axes.residual.ylabel[] == "Signed √ΔW"
        @test demo.numeric.valid == [true, true, true, true, true, false]
        @test filesize(joinpath(folder, "hetg-panels.pdf")) > 0
        @test filesize(joinpath(folder, "rgs-gaps.png")) > 0
    end
end
