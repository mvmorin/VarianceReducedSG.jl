module SmallLassoDemo

using VarianceReducedSG
using ProximalOperators
using LinearAlgebra
using Random
using Test

# Square-norm gradient
sqgrad(x) = x

function setup_data(N, n, mt)
	# Generate data
	A = randn(mt, n,N)
	y = A*randn(mt,N) # No noise

	# Normalize and reformat
	norm_consts = [norm(A[i,:]) for i = 1:size(A,1)]
	data = [A[i,:]./norm_consts[i] for i = 1:size(A,1)]
	b = [y[i]/norm_consts[i] for i = 1:length(y)]

	return data, b
end


################################################################################
# SAGA
################################################################################
@testset "Lasso" begin
	println("Lasso")
	N = 100
	n = 1000
	mt = MersenneTwister(123)

	data, b = setup_data(N,n,mt)


	f(y,x,i) = (y .= data[i].*sqgrad(dot(data[i],x) - b[i]))
	fs(x,i) = sqgrad(x - b[i])
	vrg = VRGradient(f, zeros(N), n)
	vrg_lin = LinearVRG(fs, data, zeros(N))
	vrg_uni = UniformVRG(f, zeros(N), n)
	reg = NormL1(.01)

	# Solve
	nstages = n*100
	alg = SAGA(vrg, 10*randn(mt,N), 1/3, mt, prox_f=reg)
	iterate!(alg, nstages, NoLog(), nstages)
	

	x_star = primiterates(alg)
	println(x_star)
	fval(x,y) = norm(x - x_star)
	progress = ShowFuncVal(fval, "|| x - x^*||")


	# Solve again for different params and solvers, all should give same result
	println("Prox-SAGA - VRGradient")
	nstages = n*100
	iterperlog = Int(floor(nstages/10))
	alg = SAGA(vrg, randn(mt,N), 1/10, mt, prox_f=reg)
	iterate!(alg, nstages, (progress, ShowNewLine()), iterperlog)
	@test isapprox(primiterates(alg), x_star)


	println("Prox-SAGA - LinearVRG - Importance")
	nstages = n*100
	iterperlog = Int(floor(nstages/10))
	alg = SAGA(
		vrg_lin, randn(mt,N), 1/4, mt, prox_f=reg, weights=0.5.+rand(mt,n))
	iterate!(
		alg, nstages, (progress, ShowNewLine()), iterperlog)
	@test isapprox(primiterates(alg), x_star)
	

	println("Prox-SVRG - UniformVRG")
	stagelen = n/10
	nstages = 50*n/stagelen
	iterperlog = Int(floor(nstages*stagelen/10))
	alg = SVRG(vrg_uni, randn(mt,N), 1/8, stagelen, mt, prox_f=reg)
	iterate!(
		alg, nstages, (progress, ShowNewLine()), iterperlog)
	@test isapprox(primiterates(alg), x_star)
	

	println("Prox-LSVRG - UniformVRG")
	q = 10/n
	nstages = n*50
	iterperlog = Int(floor(nstages/10))
	alg = LSVRG(
		vrg_uni, randn(mt,N), 1/8, q, mt, prox_f=reg,weights=0.5.+rand(mt,n))
	iterate!(
		alg, nstages, (progress, ShowNewLine()), iterperlog)
	@test isapprox(primiterates(alg), x_star)
end

end
