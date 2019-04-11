export SAGA, SVRG

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
	i = findfirst(c -> c >= u, s.cp) # There are probably better ways to do this
	return i, s.p[i]
end



################################################################################
# Variance Reduced Algorithms
################################################################################
abstract type VRAlgorithm end

update!(alg::VRAlgorithm) = nothing
update!(alg::VRAlgorithm, iter) = update!(alg::VRAlgorithm)
update!(alg::VRAlgorithm, iter, stage) = update!(alg::VRAlgorithm, iter)

stageupdate!(alg::VRAlgorithm) = nothing
stageupdate!(alg::VRAlgorithm, stage) = stageupdate!(alg::VRAlgorithm)

initialize!(alg::VRAlgorithm) = nothing

primiterates(alg::VRAlgorithm) = nothing
dualiterates(alg::VRAlgorithm) = nothing


##############################
# SAGA
struct SAGA{T,X,VRG,S<:IndexSampling} <: VRAlgorithm
	stepsize::T
	x::X
	x0::X
	vrg::VRG
	sampling::S
end
# Constructors
function SAGA(stepsize, x, vrg, x0, rng)
	sampling = UniformSampling(rng, nfunc(vrg))
	SAGA(stepsize, x, x0, vrg, sampling)
end
function SAGA(stepsize, x, vrg, x0, rng, w)
	sampling = WeightedSampling(rng, w)
	SAGA(stepsize, x, x0, vrg, sampling)
end
# Interface
function initialize!(alg::SAGA)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::SAGA, iter, stage)
	(i,p) = alg.sampling()
	ss_innv = alg.stepsize/(nfunc(alg.vrg)*p)
	addandstore_vrgrad!(alg.x, alg.vrg, i, -ss_innv, -alg.stepsize)
end
stageupdate!(alg::SAGA, stage) = 1
primiterates(alg::SAGA) = alg.x
dualiterates(alg::SAGA) = alg.vrg


##############################
# SVRG
struct SVRG{T,I,X,VRG,S<:IndexSampling} <: VRAlgorithm
	stepsize::T
	stagelen::I
	x::X
	vrg::VRG
	x0::X
	sampling::S
end
# Constructors
function SVRG(stepsize, stagelen, x, vrg, x0, rng)
	sampling = UniformSampling(rng, nfunc(vrg))
	SVRG(stepsize,stagelen,x,vrg,x0,sampling)
end
function SVRG(stepsize, stagelen, x, vrg, x0, rng, w)
	sampling = WeightedSampling(rng, w)
	SVRG(stepsize,stagelen,x,vrg,x0,sampling)
end
# Interface
function initialize!(alg::SVRG)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::SVRG, iter, stage)
	(i,p) = alg.sampling()
	ss_innv = alg.stepsize/(nfunc(alg.vrg)*p)
	add_vrgrad!(alg.x, alg.vrg, i, -ss_innv, -alg.stepsize)
end
function stageupdate!(alg::SVRG, stage)
	store_grad!(alg.vrg, alg.x)
	return alg.stagelen
end
primiterates(alg::SVRG) = alg.x
dualiterates(alg::SVRG) = alg.vrg
