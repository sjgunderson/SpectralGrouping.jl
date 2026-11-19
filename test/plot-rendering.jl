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
