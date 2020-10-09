export SAGA, SVRG, LSVRG, SVAG, SAG, QSAGA, ILSVRG, ASVAG, SG

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

expectedstagelen(alg::VRAlgorithm) = nothing


##############################
# SAGA
function SAGA(vrg, x0, stepsize, rng; weights=nothing, prox_f=IndFree())
	innoweight = length(vrg)
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
end
# Constructor
function SVAG(
		vrg, x0, stepsize, innoweight, rng; weights=nothing, prox_f=IndFree())

	T = eltype(x0)
	stepsize = T(stepsize)
	innoweight = T(innoweight)

	x = similar(x0)
	x_tmp = similar(x0)

	sampling = (weights == nothing) ?	UniformSampling(rng, length(vrg)) :
										WeightedSampling(rng, weights)

	SVAG(x,x_tmp,x0,vrg,prox_f,stepsize,innoweight,sampling)
end
# Interface
function initialize!(alg::SVAG)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::SVAG, iter, stage)
	(i,p) = alg.sampling()

	n = length(alg.vrg)
	innoweight_i = alg.innoweight/(n*n*p)
	vrgrad_store!(alg.x_tmp, alg.vrg, alg.x, i, innoweight_i)
	alg.x_tmp .= alg.x .- alg.stepsize.*alg.x_tmp
	prox!(alg.x, alg.prox_f, alg.x_tmp, alg.stepsize)
end
stageupdate!(alg::SVAG) = 1
primiterates(alg::SVAG) = alg.x
dualiterates(alg::SVAG) = alg.vrg
expectedstagelen(alg::SVAG) = 1



##############################
# Adaptive SVAG
struct ASVAG{X,VRG,PF,T,S,F} <: VRAlgorithm
	x::X
	x_tmp::X
	x0::X
	vrg::VRG
	prox_f::PF
	stepsize::T
	innoweight::Base.RefValue{T}
	innov::X
	expinnov::X
	decay::T
	biascorr::Base.RefValue{T}
	sampling::S
	stepcorrection::F
end
# Constructor
function ASVAG(
		vrg, x0, stepsize, rng;
		weights=nothing, prox_f=IndFree(), enforce_step=false)

	T = eltype(x0)
	stepsize = T(stepsize)
	innoweight = T(length(vrg))

	x = similar(x0)
	x_tmp = similar(x0)

	innov = similar(x0)
	expinnov = zero(x0)
	biascorr = zero(T)
	decay = T(1/10)

	sampling = (weights == nothing) ?	UniformSampling(rng, length(vrg)) :
										WeightedSampling(rng, weights)
	if enforce_step
		stepcorrection = (n,theta) -> max(
			2/(2 + abs(n-theta)),
			2/(2 + (n-theta)*(theta-1)/n*((theta-1)/n-1+sign(theta-1)*sqrt(2)) )
			)
	else
		stepcorrection = (n,theta) -> 1
	end
	ASVAG(
		x, x_tmp, x0, vrg, prox_f, stepsize, Ref(innoweight),
		innov, expinnov, decay, Ref(biascorr),
		sampling, stepcorrection
		)
end
# Interface
function initialize!(alg::ASVAG)
	reset!(alg.vrg)
	alg.x .= alg.x0
	alg.innoweight[] = length(alg.vrg)
	alg.expinnov .= zero(eltype(alg.expinnov))
	alg.biascorr[] = zero(alg.biascorr[])
end
function update!(alg::ASVAG, iter, stage)
	(i,p) = alg.sampling()
	n = length(alg.vrg)

	# Update expected innovation
	state = innov!(alg.innov, alg.vrg, alg.x, i)
	alg.expinnov .= alg.innov .+ (1-alg.decay).*alg.expinnov
	alg.biascorr[] = 1 + (1-alg.decay)*alg.biascorr[]

	# Calc innovation weight and make stepsize correction
	alg.innoweight[] =
		n*dot(alg.expinnov, alg.innov) /
		(alg.biascorr[]*norm(alg.innov)^2 + 1e-8)
	lim = n
	alg.innoweight[] = min(lim, alg.innoweight[])
	alg.innoweight[] = max(-lim, alg.innoweight[])

	innoweight_i = alg.innoweight[] / (n*n*p)
	stepsize = alg.stepsize * alg.stepcorrection(n,alg.innoweight[])

	# Take VRSG step
	grad_approx!(alg.x_tmp, alg.vrg)
	alg.x_tmp .= alg.x .- stepsize.*(innoweight_i.*alg.innov .+ alg.x_tmp)
	prox!(alg.x, alg.prox_f, alg.x_tmp, stepsize)
	store_from_innov!(alg.vrg, alg.innov, state)
end
stageupdate!(alg::ASVAG) = 1
primiterates(alg::ASVAG) = alg.x
dualiterates(alg::ASVAG) = alg.vrg
expectedstagelen(alg::ASVAG) = 1



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
end
# Constructor
function SVRG(vrg, x0, stepsize, stagelen, rng;weights=nothing,prox_f=IndFree())
	T = eltype(x0)
	stepsize = T(stepsize)

	x = similar(x0)
	x_tmp = similar(x0)

	sampling = (weights == nothing) ?	UniformSampling(rng, length(vrg)) :
										WeightedSampling(rng, weights)
	SVRG(x,x_tmp,x0,vrg,prox_f,stepsize,stagelen,sampling)
end
# Interface
function initialize!(alg::SVRG)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::SVRG, iter, stage)
	(i,p) = alg.sampling()
	innoweight = 1/(length(alg.vrg)*p)
	vrgrad!(alg.x_tmp, alg.vrg, alg.x, i, innoweight)
	alg.x_tmp .= alg.x .- alg.stepsize.*alg.x_tmp
	prox!(alg.x, alg.prox_f, alg.x_tmp, alg.stepsize)
end
function stageupdate!(alg::SVRG)
	store_from_point!(alg.vrg, alg.x)
	return alg.stagelen
end
primiterates(alg::SVRG) = alg.x
dualiterates(alg::SVRG) = alg.vrg
expectedstagelen(alg::SVRG) = alg.stagelen



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
end
# Constructor
function LSVRG(vrg, x0, stepsize, q, rng; weights=nothing,prox_f=IndFree())
	T = eltype(x0)
	stepsize = T(stepsize)

	x = similar(x0)
	x_tmp = similar(x0)

	sampling = (weights == nothing) ?	UniformSampling(rng, length(vrg)) :
										WeightedSampling(rng, weights)
	LSVRG(x,x_tmp,x0,vrg,prox_f,stepsize,q,rng,sampling)
end
# Interface
function initialize!(alg::LSVRG)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::LSVRG, iter, stage)
	(i,p) = alg.sampling()

	innoweight = 1/(length(alg.vrg)*p)
	vrgrad!(alg.x_tmp, alg.vrg, alg.x, i, innoweight)

	rand(alg.rng) < alg.q && store_from_point!(alg.vrg, alg.x)

	alg.x_tmp .= alg.x .- alg.stepsize.*alg.x_tmp
	prox!(alg.x, alg.prox_f, alg.x_tmp, alg.stepsize)
end
stageupdate!(alg::LSVRG) = 1
primiterates(alg::LSVRG) = alg.x
dualiterates(alg::LSVRG) = alg.vrg
expectedstagelen(alg::LSVRG) = 1



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
end
# Constructor
function ILSVRG(vrg, x0, stepsize, q, rng; weights=nothing,prox_f=IndFree())
	T = eltype(x0)
	stepsize = T(stepsize)

	x = similar(x0)
	x_tmp = similar(x0)

	sampling = (weights == nothing) ?	UniformSampling(rng, length(vrg)) :
										WeightedSampling(rng, weights)
	ILSVRG(x,x_tmp,x0,vrg,prox_f,stepsize,q,rng,sampling)
end
# Interface
function initialize!(alg::ILSVRG)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::ILSVRG, iter, stage)
	(i,p) = alg.sampling()
	n = length(alg.vrg)

	innoweight = 1/(n*p)
	vrgrad!(alg.x_tmp, alg.vrg, alg.x, i, innoweight)

	for j = 1:n
		rand(alg.rng) < alg.q && store_from_point!(alg.vrg, alg.x, j)
	end

	alg.x_tmp .= alg.x .- alg.stepsize.*alg.x_tmp
	prox!(alg.x, alg.prox_f, alg.x_tmp, alg.stepsize)
end
stageupdate!(alg::ILSVRG) = 1
primiterates(alg::ILSVRG) = alg.x
dualiterates(alg::ILSVRG) = alg.vrg
expectedstagelen(alg::ILSVRG) = 1



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
end
# Constructor
function QSAGA(
		vrg, x0, stepsize, q, rng;
		weights=nothing, prox_f=IndFree(), replace=true)
	T = eltype(x0)
	stepsize = T(stepsize)

	x = similar(x0)
	x_tmp = similar(x0)
	dualindex = Vector{Int}(undef,q)

	sampling = (weights == nothing) ?	UniformSampling(rng, length(vrg)) :
										WeightedSampling(rng, weights)
	QSAGA(x,x_tmp,x0,vrg,prox_f,stepsize,dualindex,replace,rng,sampling)
end
# Interface
function initialize!(alg::QSAGA)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::QSAGA, iter, stage)
	n = length(alg.vrg)
	(i,p) = alg.sampling()

	innoweight = 1/(n*p)
	vrgrad!(alg.x_tmp, alg.vrg, alg.x, i, innoweight)

	sample!(alg.rng, 1:n, alg.dualindex, replace=alg.replace)
	store_from_point!(alg.vrg, alg.x, alg.dualindex)

	alg.x_tmp .= alg.x .- alg.stepsize.*alg.x_tmp
	prox!(alg.x, alg.prox_f, alg.x_tmp, alg.stepsize)
end
stageupdate!(alg::QSAGA) = 1
primiterates(alg::QSAGA) = alg.x
dualiterates(alg::QSAGA) = alg.vrg
expectedstagelen(alg::QSAGA) = 1



##############################
# SG, ordinary stoachastic gradient, no variance reduction
struct SG{X,T,VRG,PF,SF,RNG,S} <: VRAlgorithm
	x::X
	x_tmp::X
	x0::X
	vrg::VRG
	prox_f::PF
	stepsize_f::SF
	iter::Base.RefValue{T}
	rng::RNG
	sampling::S
end
# Constructor
function SG(vrg, x0, stepsize_f, rng; weights=nothing, prox_f=IndFree())
	iter = zero(eltype(x0))

	x = similar(x0)
	x_tmp = similar(x0)

	sampling = (weights == nothing) ?
		UniformSampling(rng, length(vrg)) :
		WeightedSampling(rng, weights)

	SG(x,x_tmp,x0,vrg,prox_f,stepsize_f,Ref(iter),rng,sampling)
end
# Interface
function initialize!(alg::SG)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::SG, iter, stage)
	(i,p) = alg.sampling()

	stepsize = alg.stepsize_f(alg.iter[])
	alg.iter[] += 1

	grad!(alg.x_tmp, alg.vrg, alg.x, i)
	alg.x_tmp .= alg.x .- stepsize.*alg.x_tmp
	prox!(alg.x, alg.prox_f, alg.x_tmp, stepsize)
end
stageupdate!(alg::SG) = 1
primiterates(alg::SG) = alg.x
dualiterates(alg::SG) = alg.vrg
expectedstagelen(alg::SG) = 1
