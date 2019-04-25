export SAGA, SVRG, LSVRG

export primiterates, dualiterates

################################################################################
# Index Samplings
################################################################################
abstract type IndexSampling end


struct UniformSampling{RNG} <: IndexSampling
	rng::RNG
	n::Int
end
(s::UniformSampling)() = (rand(s.rng, 1:s.n), 1/s.n)



struct WeightedSampling{RNG, CP} <: IndexSampling
	rng::RNG
	p::CP
	cp::CP
	function WeightedSampling(rng, w)
		p = w./sum(w)
		cp = cumsum(p)
		new{typeof(rng), typeof(cp)}(rng, p, cp)
	end
end
function (s::WeightedSampling)()
	u = rand(s.rng)
	i = searchsortedfirst(s.cp, u)
	return i, s.p[i]
end



################################################################################
# Variance Reduced Algorithms
################################################################################
abstract type VRAlgorithm end

update!(alg::VRAlgorithm) = nothing
update!(alg::VRAlgorithm, iter) = update!(alg::VRAlgorithm)
update!(alg::VRAlgorithm, iter, stage) = update!(alg::VRAlgorithm, iter)

stageupdate!(alg::VRAlgorithm) =
	error("stageupdate! need to return the stage length")
stageupdate!(alg::VRAlgorithm, stage) = stageupdate!(alg::VRAlgorithm)
stageupdate!(alg::VRAlgorithm, iter, stage) =
	stageupdate!(alg::VRAlgorithm, stage)

initialize!(alg::VRAlgorithm) = nothing

primiterates(alg::VRAlgorithm) = nothing
dualiterates(alg::VRAlgorithm) = nothing


##############################
# SAGA
struct SAGA{T,X,VRG,S<:IndexSampling} <: VRAlgorithm
	stepsize::T
	x::X
	x_tmp::X
	x0::X
	vrg::VRG
	sampling::S
end
# Constructors
function SAGA(stepsize, vrg, x0, rng)
	sampling = UniformSampling(rng, nfunc(vrg))
	x = similar(x0)
	x_tmp = similar(x0)
	SAGA(stepsize, x, x_tmp, x0, vrg, sampling)
end
function SAGA(stepsize, vrg, x0, rng, w)
	sampling = WeightedSampling(rng, w)
	x = similar(x0)
	x_tmp = similar(x0)
	SAGA(stepsize, x, x_tmp, x0, vrg, sampling)
end
# Interface
function initialize!(alg::SAGA)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::SAGA, iter, stage)
	(i,p) = alg.sampling()
	innv_weight = 1/(nfunc(alg.vrg)*p)
	vrgrad_store!(alg.x_tmp, alg.vrg, alg.x, i, innv_weight)
	alg.x .-= alg.stepsize.*alg.x_tmp
end
stageupdate!(alg::SAGA) = 1
primiterates(alg::SAGA) = alg.x
dualiterates(alg::SAGA) = alg.vrg


##############################
# SVRG
struct SVRG{T,I,X,VRG,S<:IndexSampling} <: VRAlgorithm
	stepsize::T
	stagelen::I
	x::X
	x_tmp::X
	vrg::VRG
	x0::X
	sampling::S
end
# Constructors
function SVRG(stepsize, stagelen, vrg, x0, rng)
	sampling = UniformSampling(rng, nfunc(vrg))
	x = similar(x0)
	x_tmp = similar(x0)
	SVRG(stepsize,stagelen,x,x_tmp,vrg,x0,sampling)
end
function SVRG(stepsize, stagelen, vrg, x0, rng, w)
	sampling = WeightedSampling(rng, w)
	x = similar(x0)
	x_tmp = similar(x0)
	SVRG(stepsize,stagelen,x,x_tmp,vrg,x0,sampling)
end
# Interface
function initialize!(alg::SVRG)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::SVRG, iter, stage)
	(i,p) = alg.sampling()
	innv_weight = 1/(nfunc(alg.vrg)*p)
	vrgrad!(alg.x_tmp, alg.vrg, alg.x, i, innv_weight)
	alg.x .-= alg.stepsize.*alg.x_tmp
end
function stageupdate!(alg::SVRG)
	grad_store!(alg.vrg, alg.x)
	return alg.stagelen
end
primiterates(alg::SVRG) = alg.x
dualiterates(alg::SVRG) = alg.vrg



##############################
# Loopless SVRG
struct LSVRG{T,X,VRG,S<:IndexSampling,Q,RNG} <:VRAlgorithm
	stepsize::T
	x::X
	x_tmp::X
	vrg::VRG
	x0::X
	sampling::S
	q::Q
	rng::RNG
end
# Constructors
function LSVRG(stepsize, vrg, x0, q, rng)
	sampling = UniformSampling(rng, nfunc(vrg))
	x = similar(x0)
	x_tmp = similar(x0)
	LSVRG(stepsize, x, x_tmp, vrg, x0, sampling, q ,rng)
end
function LSVRG(stepsize, vrg, x0, q, rng, w)
	sampling = WeightedSampling(rng, w)
	x = similar(x0)
	x_tmp = similar(x0)
	LSVRG(stepsize, x, x_tmp, vrg, x0, sampling, q ,rng)
end
# Interface
function initialize!(alg::LSVRG)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::LSVRG, iter, stage)
	(i,p) = alg.sampling()
	innv_weight = 1/(nfunc(alg.vrg)*p)
	vrgrad!(alg.x_tmp, alg.vrg, alg.x, i, innv_weight)
	rand(alg.rng) < alg.q && grad_store!(alg.vrg, alg.x)
	alg.x .-= alg.stepsize.*alg.x_tmp
end
stageupdate!(alg::LSVRG) = 1
primiterates(alg::LSVRG) = alg.x
dualiterates(alg::LSVRG) = alg.vrg
