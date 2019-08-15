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
	niter = n*100
	alg = SAGA(vrg, 10*randn(mt,N), 1/3, mt, prox_f=reg)
	iterate!(alg, niter)
	

	x_star = primiterates(alg)
	fval(x,y) = norm(x - x_star)
	progress = ShowFuncVal(fval, "|| x - x^*||")


	# Solve again for different params and solvers, all should give same result
	println("Prox-SAGA - VRGradient")
	niter = n*100
	nlogs = 10
	alg = SAGA(vrg, randn(mt,N), 1/10, mt, prox_f=reg)
	iterate!(alg, niter, (progress, ShowNewLine()), nlogs)
	@test isapprox(primiterates(alg), x_star)


	println("Prox-SAGA - LinearVRG - Importance")
	niter = n*100
	nlogs = 10
	alg = SAGA(
		vrg_lin, randn(mt,N), 1/4, mt, prox_f=reg, weights=0.5.+rand(mt,n))
	iterate!(alg, niter, (progress, ShowNewLine()), nlogs)
	@test isapprox(primiterates(alg), x_star)
	

	println("Prox-SAG - LinearVRG - Importance")
	niter = n*100
	nlogs = 10
	alg = SAG(
		vrg_lin, randn(mt,N), 1/4, mt, prox_f=reg, weights=0.5.+rand(mt,n))
	iterate!(alg, niter, (progress, ShowNewLine()), nlogs)
	@test isapprox(primiterates(alg), x_star)


	println("Prox-SVAG - LinearVRG")
	niter = n*100
	nlogs = 10
	alg = SVAG(vrg_lin, randn(mt,N), 1/4, n/20, mt, prox_f=reg)
	iterate!(alg, niter, (progress, ShowNewLine()), nlogs)
	@test isapprox(primiterates(alg), x_star)


	println("Prox-SVRG - UniformVRG")
	stagelen = n/10
	niter = 50*n
	nlogs = 10
	alg = SVRG(vrg_uni, randn(mt,N), 1/8, stagelen, mt, prox_f=reg)
	iterate!(alg, niter, (progress, ShowNewLine()), nlogs)
	@test isapprox(primiterates(alg), x_star)
	

	println("Prox-LSVRG - UniformVRG - Importance")
	q = 10/n
	niter = n*50
	nlogs = 10
	alg = LSVRG(
		vrg_uni, randn(mt,N), 1/8, q, mt, prox_f=reg,weights=0.5.+rand(mt,n))
	iterate!(alg, niter, (progress, ShowNewLine()), nlogs)
	@test isapprox(primiterates(alg), x_star)


	println("Prox-ILSVRG - VRGradient - Importance")
	q = 10/n
	niter = n*50
	nlogs = 10
	alg = ILSVRG(
		vrg, randn(mt,N), 1/8, q, mt, prox_f=reg,weights=0.5.+rand(mt,n))
	iterate!(alg, niter, (progress, ShowNewLine()), nlogs)
	@test isapprox(primiterates(alg), x_star)


	println("Prox-QSAGA - LinearVRG - q = 10")
	q = 10
	niter = n*50
	nlogs = 10
	alg = QSAGA(
		vrg_lin, randn(mt,N), 1/8, q, mt, prox_f=reg)
	iterate!(alg, niter, (progress, ShowNewLine()), nlogs)
	@test isapprox(primiterates(alg), x_star)


	println("Prox-QSAGA - VRGadient - q = 10 - Without replacement")
	q = 10
	niter = n*50
	nlogs = 10
	alg = QSAGA(
		vrg, randn(mt,N), 1/8, q, mt, prox_f=reg, replace=false)
	iterate!(alg, niter, (progress, ShowNewLine()), nlogs)
	@test isapprox(primiterates(alg), x_star)
end

end
