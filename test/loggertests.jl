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
	VarianceReducedSG.grad_store!(vrg, x0)

	alg = SAGA(vrg,x0,1.0,MersenneTwister(800))
	alg.x .= x0

	f1(x,y) = x[1]
	f2(x,y) = length(y.y)

	f1_cache = CacheFuncVal(f1)
	f2_cache = CacheFuncVal(f2,Int)

	nlogs = 23
	niter = 23
	logiters = zeros(Int,nlogs)
	f1_vals = zeros(nlogs)
	f2_vals = zeros(Int,nlogs)

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
		ShowNewLine("\nEnd of log.\n"),
		StoreFuncVal(f1_cache, f1_vals),
		StoreFuncVal(f2_cache, f2_vals),
		StoreLogIterations(logiters),
		)

	iterate!(alg, niter, loggers, nlogs, warmstart=true)

	loginterval = Int(ceil(niter/nlogs))
	@test all(logiters .== (0:loginterval:niter)[1:nlogs])
	@test all(isapprox(f1_vals, x0 .+ (0:loginterval:niter)[1:nlogs]))
	@test all(f2_vals .== n)
end

test_loggers()

end
