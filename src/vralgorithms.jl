export SAGA, SVRG, LSVRG, SVAG, SAG, QSAGA, ILSVRG

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

exp_stagelen(alg::VRAlgorithm) = nothing


##############################
# SAGA
function SAGA(vrg, x0, stepsize, rng; weights=nothing, prox_f=IndFree())
	innoweight = nfunc(vrg)
	SVAG(
		vrg, x0, stepsize, innoweight, rng, weights=weights, prox_f=prox_f)
end



##############################
# SAG
function SAG(vrg, x0, stepsize, rng; weights=nothing, prox_f=IndFree())
	innoweight = 1
	SVAG(
		vrg, x0, stepsize, innoweight, rng, weights=weights, prox_f=prox_f)
end



##############################
# SVAG
struct SVAG{X,VRG,PF,T,S} <: VRAlgorithm
	x::X
	x_tmp::X
	x0::X
	vrg::VRG
	prox_f::PF
	stepsize::T
	innoweight::T
	sampling::S

	function SVAG(vrg, prox_f, x0, stepsize, innoweight, sampling)
		x = similar(x0)
		x_tmp = similar(x0)
		X,VRG,PF,T,S = typeof.((x,vrg,prox_f,stepsize,sampling))
		new{X,VRG,PF,T,S}(x,x_tmp,x0,vrg,prox_f,stepsize,innoweight,sampling)
	end
end
# Constructors
function SVAG(
		vrg, x0, stepsize, innoweight, rng; weights=nothing, prox_f=IndFree())

	sampling = (weights == nothing) ?	UniformSampling(rng, nfunc(vrg)) :
										WeightedSampling(rng, weights)
	SVAG(vrg, prox_f, x0, stepsize, innoweight, sampling)
end
# Interface
function initialize!(alg::SVAG)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::SVAG, iter, stage)
	(i,p) = alg.sampling()

	n = nfunc(alg.vrg)
	sample_innoweight = alg.innoweight/(n*n*p)
	vrgrad_store!(alg.x_tmp, alg.vrg, alg.x, i, sample_innoweight)
	alg.x_tmp .= alg.x .- alg.stepsize.*alg.x_tmp
	prox!(alg.x, alg.prox_f, alg.x_tmp, alg.stepsize)
end
stageupdate!(alg::SVAG) = 1
primiterates(alg::SVAG) = alg.x
dualiterates(alg::SVAG) = alg.vrg
exp_stagelen(alg::SVAG) = 1



##############################
# SVRG
struct SVRG{X,VRG,PF,T,I,S} <: VRAlgorithm
	x::X
	x_tmp::X
	x0::X
	vrg::VRG
	prox_f::PF
	stepsize::T
	stagelen::I
	sampling::S

	function SVRG(vrg,prox_f,x0,stepsize,stagelen,sampling)
		x = similar(x0)
		x_tmp = similar(x0)
		X,VRG,PF,T,I,S = typeof.((x,vrg,prox_f,stepsize,stagelen,sampling))
		new{X,VRG,PF,T,I,S}(x,x_tmp,x0,vrg,prox_f,stepsize,stagelen,sampling)
	end
end
# Constructors
function SVRG(vrg, x0, stepsize, stagelen, rng;weights=nothing,prox_f=IndFree())
	sampling = (weights == nothing) ?	UniformSampling(rng, nfunc(vrg)) :
										WeightedSampling(rng, weights)
	SVRG(vrg, prox_f, x0, stepsize, stagelen, sampling)
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
	alg.x_tmp .= alg.x .- alg.stepsize.*alg.x_tmp
	prox!(alg.x, alg.prox_f, alg.x_tmp, alg.stepsize)
end
function stageupdate!(alg::SVRG)
	grad_store!(alg.vrg, alg.x)
	return alg.stagelen
end
primiterates(alg::SVRG) = alg.x
dualiterates(alg::SVRG) = alg.vrg
exp_stagelen(alg::SVRG) = alg.stagelen



##############################
# Loopless SVRG
struct LSVRG{X,VRG,PF,T,Q,RNG,S} <:VRAlgorithm
	x::X
	x_tmp::X
	x0::X
	vrg::VRG
	prox_f::PF
	stepsize::T
	q::Q
	rng::RNG
	sampling::S

	function LSVRG(vrg,prox_f,x0,stepsize,q,rng,sampling)
		x = similar(x0)
		x_tmp = similar(x0)
		X,VRG,PF,T,Q,RNG,S = typeof.((x,vrg,prox_f,stepsize,q,rng,sampling))
		new{X,VRG,PF,T,Q,RNG,S}(x,x_tmp,x0,vrg,prox_f,stepsize,q,rng,sampling)
	end
end
# Constructors
function LSVRG(vrg, x0, stepsize, q, rng; weights=nothing,prox_f=IndFree())
	sampling = (weights == nothing) ?	UniformSampling(rng, nfunc(vrg)) :
										WeightedSampling(rng, weights)
	LSVRG(vrg, prox_f, x0, stepsize, q, rng, sampling)
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

	alg.x_tmp .= alg.x .- alg.stepsize.*alg.x_tmp
	prox!(alg.x, alg.prox_f, alg.x_tmp, alg.stepsize)
end
stageupdate!(alg::LSVRG) = 1
primiterates(alg::LSVRG) = alg.x
dualiterates(alg::LSVRG) = alg.vrg
exp_stagelen(alg::LSVRG) = 1



##############################
# Incoherent Loopless SVRG
struct ILSVRG{X,VRG,PF,T,Q,RNG,S} <:VRAlgorithm
	x::X
	x_tmp::X
	x0::X
	vrg::VRG
	prox_f::PF
	stepsize::T
	q::Q
	rng::RNG
	sampling::S

	function ILSVRG(vrg,prox_f,x0,stepsize,q,rng,sampling)
		x = similar(x0)
		x_tmp = similar(x0)
		X,VRG,PF,T,Q,RNG,S = typeof.((x,vrg,prox_f,stepsize,q,rng,sampling))
		new{X,VRG,PF,T,Q,RNG,S}(x,x_tmp,x0,vrg,prox_f,stepsize,q,rng,sampling)
	end
end
# Constructors
function ILSVRG(vrg, x0, stepsize, q, rng; weights=nothing,prox_f=IndFree())
	sampling = (weights == nothing) ?	UniformSampling(rng, nfunc(vrg)) :
										WeightedSampling(rng, weights)
	ILSVRG(vrg, prox_f, x0, stepsize, q, rng, sampling)
end
# Interface
function initialize!(alg::ILSVRG)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::ILSVRG, iter, stage)
	(i,p) = alg.sampling()
	n = nfunc(alg.vrg)

	innv_weight = 1/(n*p)
	vrgrad!(alg.x_tmp, alg.vrg, alg.x, i, innv_weight)

	for j = 1:n
		rand(alg.rng) < alg.q && grad_store!(alg.vrg, alg.x, j)
	end

	alg.x_tmp .= alg.x .- alg.stepsize.*alg.x_tmp
	prox!(alg.x, alg.prox_f, alg.x_tmp, alg.stepsize)
end
stageupdate!(alg::ILSVRG) = 1
primiterates(alg::ILSVRG) = alg.x
dualiterates(alg::ILSVRG) = alg.vrg
exp_stagelen(alg::ILSVRG) = 1



##############################
# qSAGA with or without replacement in the dual update
struct QSAGA{X,VRG,PF,T,RNG,S} <:VRAlgorithm
	x::X
	x_tmp::X
	x0::X
	vrg::VRG
	prox_f::PF
	stepsize::T
	dualindex::Vector{Int}
	replace::Bool
	rng::RNG
	sampling::S

	function QSAGA(vrg,prox_f,x0,stepsize,q,replace,rng,sampling)
		x = similar(x0)
		x_tmp = similar(x0)
		dualindex = Vector{Int}(undef,q)

		X,VRG,PF,T,RNG,S = typeof.((x,vrg,prox_f,stepsize,rng,sampling))
		new{X,VRG,PF,T,RNG,S}(
			x,x_tmp,x0,vrg,prox_f,stepsize,dualindex,replace,rng,sampling)
	end
end
# Constructors
function QSAGA(
		vrg, x0, stepsize, q, rng;
		weights=nothing, prox_f=IndFree(), replace=true)
	sampling = (weights == nothing) ?	UniformSampling(rng, nfunc(vrg)) :
										WeightedSampling(rng, weights)
	QSAGA(vrg, prox_f, x0, stepsize, q, replace, rng, sampling)
end
# Interface
function initialize!(alg::QSAGA)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::QSAGA, iter, stage)
	n = nfunc(alg.vrg)
	(i,p) = alg.sampling()

	innv_weight = 1/(n*p)
	vrgrad!(alg.x_tmp, alg.vrg, alg.x, i, innv_weight)

	sample!(alg.rng, 1:n, alg.dualindex, replace=alg.replace)
	grad_store!(alg.vrg, alg.x, alg.dualindex)

	alg.x_tmp .= alg.x .- alg.stepsize.*alg.x_tmp
	prox!(alg.x, alg.prox_f, alg.x_tmp, alg.stepsize)
end
stageupdate!(alg::QSAGA) = 1
primiterates(alg::QSAGA) = alg.x
dualiterates(alg::QSAGA) = alg.vrg
exp_stagelen(alg::QSAGA) = 1
