export SAGA, SVRG, LSVRG

export primiterates, dualiterates
export UniformSampling, nfunc

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
stageupdate!(alg::SAGA) = 1
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
function stageupdate!(alg::SVRG)
	store_grad!(alg.vrg, alg.x)
	return alg.stagelen
end
primiterates(alg::SVRG) = alg.x
dualiterates(alg::SVRG) = alg.vrg



##############################
# Loopless SVRG
struct LSVRG{T,I,X,VRG,S<:IndexSampling,SS,RNG} <:VRAlgorithm
	stepsize::T
	x::I
	x_prev::X
	vrg::VRG
	x0::X
	sampling::S
	stagesampling::SS
	rng::RNG
end
# Constructors
function LSVRG(stepsize, x, vrg, x0, q, rng)
	sampling = UniformSampling(rng, nfunc(vrg))
	LSVRG(stepsize, x, vrg, x0, sampling, q ,rng)
end
function LSVRG(stepsize, x, vrg, x0, q, rng, w)
	sampling = WeightedSampling(rng, w)
	LSVRG(stepsize, x, vrg, x0, sampling, q ,rng)
end
function LSVRG(stepsize, x, vrg, x0, sampling::IndexSampling, q, rng)
	# Stage lengths are geometrically distributed
	stagesampling = Geometric(q)
	x_prev = similar(x)
	LSVRG(stepsize, x, x_prev, vrg, x0, sampling, stagesampling, rng)
end
# Interface
function initialize!(alg::LSVRG)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::LSVRG, iter, stage)
	(i,p) = alg.sampling()
	ss_innv = alg.stepsize/(nfunc(alg.vrg)*p)
	alg.x_prev .= alg.x
	add_vrgrad!(alg.x, alg.vrg, i, -ss_innv, -alg.stepsize)
end
function stageupdate!(alg::LSVRG)
	# Note: Update from previous iterate, otherwise it is an unbounded S2GD
	store_grad!(alg.vrg, alg.x_prev)
	stagelen = rand(alg.rng, alg.stagesampling) + 1
	return stagelen
end
primiterates(alg::LSVRG) = alg.x
dualiterates(alg::LSVRG) = alg.vrg
