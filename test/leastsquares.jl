module SmallLeastSquaresDemo

using VarianceReducedSG
using LinearAlgebra
using Random
using Test


# Square-norm gradient
sqgrad(x) = x

function setup_ls(N, n, mt)
	# Generate data
	A = randn(mt, n,N)
	x_star = randn(mt, N)

	y = A*x_star # No noise

	# Normalize and reformat
	norm_consts = [norm(A[i,:]) for i = 1:size(A,1)]
	data = [A[i,:]./norm_consts[i] for i = 1:size(A,1)]
	b = [y[i]/norm_consts[i] for i = 1:length(y)]

	# Initial guess
	x0 = randn(mt, N)

	return data, b, x0, x_star
end

function solve_and_test(alg, niter, loggers, nlogs, x_star)
	iterate!(alg, niter, loggers, nlogs)
	@test isapprox(primiterates(alg), x_star)

	iterate!(alg, niter)
	@test isapprox(primiterates(alg), x_star)

	iterate!(alg, niter, loggers, nlogs)
	@test isapprox(primiterates(alg), x_star)
end


################################################################################
# SAGA, SAG, SVAG
################################################################################
function SAGlike_test()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x0, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(x,i) = sqgrad(x - b[i])
	f!(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = VRGradient(f!, zeros(N), n)
	vrg_lin = LinearVRG(f, data, zeros(N))
	
	# Create Algorithms
	algs = [
		(SAGA(vrg, x0, 1/3, mt), "SAGA - VRGradient"),
		(SAGA(vrg_lin, x0, 1/3, mt), "SAGA - LinearVRG"),
		(SAGA(vrg_lin, x0, 1/3, mt, weights=.1 .+rand(mt,n)), "SAGA - LinearVRG - Importance"),
		(SAG(vrg, x0, 1/3, mt), "SAG - VRGradient"),
		(SAG(vrg_lin, x0, 1/3, mt), "SAG - LinearVRG"),
		(SAG(vrg, x0, 1/3, mt, weights=.1 .+rand(mt,n)), "SAG - VRGradient - Importance"),
		(SVAG(vrg, x0, 1/3, n/10, mt), "SVAG - VRGradient"),
		(SVAG(vrg_lin, x0, 1/3, n/10, mt), "SVAG - LinearVRG"),
		(SVAG(vrg_lin, x0, 1/3, n/10, mt, weights=.1 .+rand(mt,n)), "SVAG - LinearVRG - Importance"),
		(ASVAG(vrg, x0, 1/3, mt), "ASVAG - VRGradient"),
		(ASVAG(vrg_lin, x0, 1/3, mt), "ASVAG - LinearVRG"),
		(ASVAG(vrg_lin, x0, 1/3, mt, weights=.1 .+rand(mt,n)), "ASVAG - LinearVRG - Importance"),
		(ASVAG(vrg_lin, x0, 1/3, mt, enforce_step=true), "ASVAG - LinearVRG - Theoretical stepsize enforced"),
		]

	for (alg, descr) in algs
		# Create Loggers
		fprog(x,y) = norm(x - x_star)
		loggers = (
			ShowTime(),
			ShowFuncVal(fprog, "|| x - x^*||"),
			ShowNewLine("...")
			)

		println(descr)

		# Solve
		niter = n*500
		nlogs = 10
		solve_and_test(alg, niter, loggers, nlogs, x_star)
	end
end


################################################################################
# SVRG
################################################################################
function SVRG_test()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x0, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(x,i) = sqgrad(x - b[i])
	f!(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = VRGradient(f!, zeros(N), n)
	vrg_lin = LinearVRG(f, data, zeros(N))
	vrg_uni = UniformVRG(f!, zeros(N), n)
	
	# Create Algorithm
	stagelen = Int(ceil(n/2))
	algs = [
		(SVRG(vrg, x0, 1/3, stagelen, mt), "SVRG - VRGradient"),
		(SVRG(vrg_uni, x0, 1/3, stagelen, mt), "SVRG - UniformVRG"),
		(SVRG(vrg_lin, x0, 1/3, stagelen, mt), "SVRG - LinearVRG"),
		(SVRG(vrg_uni, x0, 1/3, stagelen, mt, weights=.1 .+rand(mt,n)), "SVRG - UniformVRG - Importance"),
		]

	for (alg, descr) in algs
		# Create Loggers
		fprog(x,y) = norm(x - x_star)
		loggers = (
			ShowFuncVal(fprog, "|| x - x^*||"),
			ShowIterations(),
			ShowNewLine()
			)

		println(descr)

		# Solve
		niter = 100*stagelen
		nlogs = 10
		solve_and_test(alg, niter, loggers, nlogs, x_star)
	end
end


################################################################################
# Loopless SVRG
################################################################################
function LSVRG_test()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x0, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(x,i) = sqgrad(x - b[i])
	f!(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = VRGradient(f!, zeros(N), n)
	vrg_lin = LinearVRG(f, data, zeros(N))
	vrg_uni = UniformVRG(f!, zeros(N), n)

	q = 10/n # Expected update frequency
	algs = [
		(LSVRG(vrg, x0, 1/3, q, mt), "LSVRG - VRGradient"),
		(LSVRG(vrg_uni, x0, 1/3, q, mt), "LSVRG - UniformVRG"),
		(LSVRG(vrg_lin, x0, 1/3, q, mt), "LSVRG - LinearVRG"),
		(LSVRG(vrg, x0, 1/3, q, mt, weights=.1 .+ rand(mt,n)), "LSVRG - VRGradient - Importance"),
		]

	for (alg, descr) in algs
		# Create Loggers
		fprog(x,y) = norm(x - x_star)
		loggers = (
			ShowIterations(),
			ShowFuncVal(fprog, "|| x - x^*||"),
			ShowNewLine()
			)

		println(descr)

		# Setup iterations counts
		niter = 1000/q
		nlogs = 10

		# Create Algorithm and solve
		solve_and_test(alg, niter, loggers, nlogs, x_star)
	end

	# Solve again for smaller expected update frequency
	q = .1/n
	niter = 100/q
	nlogs = 10

	# Create Loggers
	fprog(x,y) = norm(x - x_star)
	loggers = (
		ShowIterations(),
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowNewLine()
		)
	println("LSVRG - Low update frequency")
	alg = LSVRG(vrg, x0, 1/3, q, mt)
	solve_and_test(alg, niter, loggers, nlogs, x_star)
end



################################################################################
# Incoherent Loopless SVRG
################################################################################
function ILSVRG_test()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x0, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(x,i) = sqgrad(x - b[i])
	f!(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = VRGradient(f!, zeros(N), n)
	vrg_lin = LinearVRG(f, data, zeros(N))
	vrg_uni = UniformVRG(f!, zeros(N), n)

	q = 10/n # Expected update frequency
	algs = [
		(ILSVRG(vrg, x0, 1/3, q, mt), "ILSVRG - VRGradient"),
		(ILSVRG(vrg_lin, x0, 1/3, q/10, mt), "ILSVRG - LinearVRG"),
		(ILSVRG(vrg, x0, 1/3, 2*q, mt, weights=.1 .+ rand(mt,n)), "ILSVRG - VRGradient - Importance"),
		]

	for (alg, descr) in algs
		# Create Loggers
		fprog(x,y) = norm(x - x_star)
		loggers = (
			ShowIterations(),
			ShowFuncVal(fprog, "|| x - x^*||"),
			ShowNewLine()
			)

		println(descr)

		# Setup iterations counts
		niter = 1000/q
		nlogs = 10

		# Create Algorithm and solve
		solve_and_test(alg, niter, loggers, nlogs, x_star)
	end
end



################################################################################
# qSAGA
################################################################################
function QSAGA_test()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x0, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(x,i) = sqgrad(x - b[i])
	f!(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = VRGradient(f!, zeros(N), n)
	vrg_lin = LinearVRG(f, data, zeros(N))

	algs = [
		(QSAGA(vrg, x0, 1/3, 1, mt), "QSAGA - VRGradient - q=1"),
		(QSAGA(vrg_lin, x0, 1/3, 10, mt), "QSAGA - LinearVRG, q=10"),
		(QSAGA(vrg, x0, 1/3, 10, mt, weights=.1 .+ rand(mt,n)), "LSVRG - VRGradient - Importance - q=10"),
		]

	for (alg, descr) in algs
		# Create Loggers
		fprog(x,y) = norm(x - x_star)
		loggers = (
			ShowIterations(),
			ShowFuncVal(fprog, "|| x - x^*||"),
			ShowNewLine()
			)

		println(descr)

		# Setup iterations counts
		niter = 100*n
		nlogs = 10

		# Create Algorithm and solve
		solve_and_test(alg, niter, loggers, nlogs, x_star)
	end

	# Solve again but without replacement in the dual update
	niter = 50*n
	nlogs = 10

	# Create Loggers
	fprog(x,y) = norm(x - x_star)
	loggers = (
		ShowIterations(),
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowNewLine()
		)
	println("LSVRG - q = 20 - Without replacement")
	alg = QSAGA(vrg, x0, 1/3, 20, mt, replace=false)
	solve_and_test(alg, niter, loggers, nlogs, x_star)
	println("LSVRG - q = 20 - With replacement")
	alg = QSAGA(vrg, x0, 1/3, 20, mt, replace=true)
	solve_and_test(alg, niter, loggers, nlogs, x_star)
end


################################################################################
# Testsets
################################################################################
@testset "Least Squares" begin
	SAGlike_test()
	SVRG_test()
	LSVRG_test()
	ILSVRG_test()
	QSAGA_test()
end

end
