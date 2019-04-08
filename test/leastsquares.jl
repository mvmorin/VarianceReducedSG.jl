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
	return data, b, x_star
end

################################################################################
# SAGA
################################################################################
function SAGA_VRGradient()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = VRGradient(f, zeros(N), n)
	
	# Primal iterates and initial value
	x = zeros(N)
	x0 = randn(mt, N)

	# Create Algorithm
	alg = SAGA(1/3, mt, x, x0, vrg)

	# Create Loggers
	fprog(x,y) = norm(x - x_star)

	loggers = (
		ShowTime(),
		ShowIterations(),
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowNewline("EOL")
		)

	# Solve
	nstages = n*500
	iterperlog = Int(floor(nstages/10))
	iterate!(alg, nstages, loggers, iterperlog)
	@test isapprox(primiterates(alg), x_star)
	
	iterate!(alg, nstages, NoLog(), iterperlog)
	@test isapprox(primiterates(alg), x_star)
	
	iterate!(alg, nstages, loggers, iterperlog)
	@test isapprox(primiterates(alg), x_star)
end

function SAGA_LinearVRG()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(x,i) = sqgrad(x - b[i])

	# VR gradient
	vrg = LinearVRG(f, data, zeros(N))
	
	# Primal iterates and initial value
	x = zeros(N)
	x0 = randn(mt, N)

	# Create Algorithm
	alg = SAGA(1/3, mt, x, x0, vrg)

	# Create Loggers
	fprog(x,y) = norm(x - x_star)

	loggers = (
		ShowIterations(),
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowNewline("EOL")
		)

	# Solve
	nstages = n*500
	iterperlog = Int(floor(nstages/10))
	iterate!(alg, nstages, loggers, iterperlog)
	@test isapprox(primiterates(alg), x_star)
	
	iterate!(alg, nstages, NoLog(), iterperlog)
	@test isapprox(primiterates(alg), x_star)
	
	iterate!(alg, nstages, loggers, iterperlog)
	@test isapprox(primiterates(alg), x_star)
end

@testset "SAGA - Least Squares" begin
	SAGA_VRGradient()
	SAGA_LinearVRG()
end


################################################################################
# SVRG
################################################################################
function SVRG_VRGradient()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = VRGradient(f, zeros(N), n)
	
	# Primal iterates and initial value
	x = zeros(N)
	x0 = randn(mt, N)

	# Create Algorithm
	alg = SVRG(1/3, Int(ceil(n/2)), mt, x, x0, vrg)

	# Create Loggers
	fprog(x,y) = norm(x - x_star)

	loggers = (
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowNewline("EOL")
		)

	# Solve
	nstages = 100
	iterperlog = Int(floor(nstages*n/2/10))
	iterate!(alg, nstages, loggers, iterperlog)
	@test isapprox(primiterates(alg), x_star)

	iterate!(alg, nstages)
	@test isapprox(primiterates(alg), x_star)
	
	iterate!(alg, nstages, loggers, iterperlog)
	@test isapprox(primiterates(alg), x_star)
end

function SVRG_UniformVRG()
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b, x_star = setup_ls(N,n,mt)

	# Gradients of square cost
	f(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))

	# VR gradient
	vrg = UniformVRG(f, zeros(N), n)
	
	# Primal iterates and initial value
	x = zeros(N)
	x0 = randn(mt, N)

	# Create Algorithm
	alg = SVRG(1/3, Int(ceil(n/2)), mt, x, x0, vrg)

	# Create Loggers
	fprog(x,y) = norm(x - x_star)

	loggers = (
		ShowFuncVal(fprog, "|| x - x^*||"),
		ShowNewline("EOL")
		)

	# Solve
	nstages = 100
	iterperlog = Int(floor(nstages*n/2/10))
	iterate!(alg, nstages, loggers, iterperlog)
	@test isapprox(primiterates(alg), x_star)
	
	iterate!(alg, nstages)
	@test isapprox(primiterates(alg), x_star)
	
	iterate!(alg, nstages, loggers, iterperlog)
	@test isapprox(primiterates(alg), x_star)
end

@testset "SVRG - Least Squares" begin
	SVRG_VRGradient()
	SVRG_UniformVRG()
end

end