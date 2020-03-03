module LoggerTests

using VarianceReducedSG
using Random
using Test

function test_loggers()
	mt = MersenneTwister(741)

	x0 = randn(mt,1)
	n = 10

	grad!(out,x,i) = out .= -1.0
	vrg = VRGradient(grad!, [0.0], n)
	VarianceReducedSG.store_from_point!(vrg, x0)

	alg = SAGA(vrg,x0,1.0,MersenneTwister(800))
	alg.x .= x0

	f1(x,y) = x[1]
	f2(x,y) = length(y.y)

	f1_cache = CacheFuncVal(f1)
	f2_cache = CacheFuncVal(f2,Int)

	nlogs = 23
	niter = 97
	extralogs = 10
	logiters = zeros(Int,nlogs + extralogs)
	f1_vals = zeros(nlogs + extralogs)
	f2_vals = zeros(Int,nlogs + extralogs)
	x0_vals = zeros(Float64,nlogs + extralogs)

	loggers = (
		ShowNewLine("Begining of log:"),
		NoLog(),
		ShowIterations(),
		ShowTime(),
		ShowNewLine(),
		f1_cache,
		f2_cache,
		ShowFuncVal(f1_cache, "Iterate"),
		ShowFuncVal(f2_cache, "Number of functions"),
		ShowAlgState(alg->alg.stepsize, "Stepsize"),
		ShowAlgState(alg->alg.x0, "Initial point"),
		StoreAlgState(alg->alg.x0[], x0_vals),
		ShowNewLine("\nEnd of log.\n"),
		StoreFuncVal(f1_cache, f1_vals),
		StoreFuncVal(f2_cache, f2_vals),
		StoreLogIterations(logiters),
		)

	iterate!(alg, niter, loggers, nlogs, warmstart=true)

	loginterval = Int(ceil(niter/nlogs))
	correct_logiters = range(0,step=loginterval,length=nlogs)
	@test all(logiters[1:nlogs] .== correct_logiters)
	@test all(logiters[(nlogs+1):end] .== 0)
	@test all(isapprox.(f1_vals[1:nlogs], x0 .+ correct_logiters))
	@test all(isapprox.(f1_vals[nlogs+1:end], 0.0))
	@test all(f2_vals[1:nlogs] .== n)
	@test all(f2_vals[nlogs+1:end] .== 0)
	@test all(x0_vals[1:nlogs] .== x0)
	@test all(x0_vals[nlogs+1:end] .== 0)
end

test_loggers()

end
