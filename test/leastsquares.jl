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

function solve_and_test(alg, nstages, loggers, iterperlog, x_star)
	iterate!(alg, nstages, loggers, iterperlog)
	@test isapprox(primiterates(alg), x_star)

	iterate!(alg, nstages, NoLog(), iterperlog)
	@test isapprox(primiterates(alg), x_star)

	iterate!(alg, nstages, loggers, iterperlog)
	@test isapprox(primiterates(alg), x_star)
end

################################################################################
# SAGA
################################################################################
function SAGA_VRGradient()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x0, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = VRGradient(f, zeros(N), n)
	
	# Create Algorithm
	alg = SAGA(vrg, x0, 1/3, mt)

	# Create Loggers
	fprog(x,y) = norm(x - x_star)

	loggers = (
		ShowTime(),
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowNewLine("SAGA - VRGradient")
		)

	# Solve
	nstages = n*500
	iterperlog = Int(floor(nstages/10))
	solve_and_test(alg, nstages, loggers, iterperlog, x_star)
end

function SAGA_LinearVRG()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x0, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(x,i) = sqgrad(x - b[i])

	# VR gradient
	vrg = LinearVRG(f, data, zeros(N))
	
	# Create Algorithm
	alg = SAGA(vrg, x0, 1/3, mt)

	# Create Loggers
	fprog(x,y) = norm(x - x_star)

	loggers = (
		ShowIterations(),
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowNewLine("SAGA - LinearVRG")
		)

	# Solve
	nstages = n*500
	iterperlog = Int(floor(nstages/10))
	solve_and_test(alg, nstages, loggers, iterperlog, x_star)
end

function SAGA_VRGradient_Importance()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x0, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = VRGradient(f, zeros(N), n)
	
	# Randomize a sample weighting
	w = .1 .+ rand(mt,n)

	# Create Algorithm
	alg = SAGA(vrg, x0, 1/3, mt, weights=w)

	# Create Loggers
	fprog(x,y) = norm(x - x_star)

	loggers = (
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowNewLine("SAGA - VRGradient - Importance")
		)

	# Solve
	nstages = n*500
	iterperlog = Int(floor(nstages/10))
	solve_and_test(alg, nstages, loggers, iterperlog, x_star)
end




################################################################################
# SVRG
################################################################################
function SVRG_VRGradient()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x0, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = VRGradient(f, zeros(N), n)
	
	# Create Algorithm
	alg = SVRG(vrg, x0, 1/3, Int(ceil(n/2)), mt)

	# Create Loggers
	fprog(x,y) = norm(x - x_star)

	loggers = (
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowNewLine("SVRG")
		)

	# Solve
	nstages = 100
	iterperlog = Int(floor(nstages*n/2/10))
	solve_and_test(alg, nstages, loggers, iterperlog, x_star)
end

function SVRG_UniformVRG()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x0, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = UniformVRG(f, zeros(N), n)
	
	# Create Algorithm
	alg = SVRG(vrg, x0, 1/3, Int(ceil(n/2)), mt)

	# Create Loggers
	fprog(x,y) = norm(x - x_star)

	loggers = (
		ShowIterations(),
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowNewLine("SVRG - UniformVRG")
		)

	# Solve
	nstages = 100
	iterperlog = Int(floor(nstages*n/2/10))
	solve_and_test(alg, nstages, loggers, iterperlog, x_star)
end

function SVRG_UniformVRG_Importance()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x0, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = UniformVRG(f, zeros(N), n)
	
	# Randomize a sample weighting
	w = .1 .+ rand(mt,n)
	
	# Create Algorithm
	alg = SVRG(vrg, x0, 1/3, Int(ceil(n/2)), mt, weights=w)

	# Create Loggers
	fprog(x,y) = norm(x - x_star)

	loggers = (
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowNewLine("SVRG - UniformVRG - Importance")
		)

	# Solve
	nstages = 100
	iterperlog = Int(floor(nstages*n/2/10))
	solve_and_test(alg, nstages, loggers, iterperlog, x_star)
end


################################################################################
# Loopless SVRG
################################################################################
function LSVRG_VRGradient()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x0, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = VRGradient(f, zeros(N), n)

	# Create Loggers
	fprog(x,y) = norm(x - x_star)

	loggers = (
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowTime(),
		ShowNewLine("LSVRG - VRGradient")
		)

	# Setup iterations counts
	q = 10/n # Expected update frequency
	nstages = 1000/q
	iterperlog = Int(floor(nstages/10))

	# Create Algorithm and solve
	alg = LSVRG(vrg, x0, 1/3, q, mt)
	solve_and_test(alg, nstages, loggers, iterperlog, x_star)

	# Solve again for smaller expected update frequency
	q = .1/n
	nstages = 100/q
	iterperlog = Int(floor(nstages/10))

	alg = LSVRG(vrg, x0, 1/3, q, mt)
	solve_and_test(alg, nstages, loggers, iterperlog, x_star)
end

function LSVRG_UniformVRG_Importance()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x0, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = UniformVRG(f, zeros(N), n)

	# Create Loggers
	fprog(x,y) = norm(x - x_star)

	loggers = (
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowTime(),
		ShowNewLine("LSVRG - UnifromVRG - Importance")
		)

	# Randomize a sample weighting
	w = .1 .+ rand(mt,n)

	# Setup iterations counts
	q = 10/n # Expected update frequency
	nstages = 1000/q
	iterperlog = Int(floor(nstages/10))

	# Create Algorithm and solve
	alg = LSVRG(vrg, x0, 1/3, q, mt, weights=w)
	solve_and_test(alg, nstages, loggers, iterperlog, x_star)

	# Solve again for smaller expected update frequency
	q = .1/n
	nstages = 100/q
	iterperlog = Int(floor(nstages/10))

	alg = LSVRG(vrg, x0, 1/3, q, mt, weights=w)
	solve_and_test(alg, nstages, loggers, iterperlog, x_star)
end

################################################################################
# Testsets
################################################################################
@testset "SAGA - Least Squares" begin
	SAGA_VRGradient()
	SAGA_LinearVRG()
	SAGA_VRGradient_Importance()
end

@testset "SVRG - Least Squares" begin
	SVRG_VRGradient()
	SVRG_UniformVRG()
	SVRG_UniformVRG_Importance()
end

@testset "Loopless SVRG - Least Squares" begin
	LSVRG_VRGradient()
	LSVRG_UniformVRG_Importance()
end

end
