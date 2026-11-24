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
