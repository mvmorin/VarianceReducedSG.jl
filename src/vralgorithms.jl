export SAGA, SVRG

export primiterates, dualiterates

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
struct SAGA{T, RNG, X, VRG} <: VRAlgorithm
	stepsize::T
	rng::RNG
	x::X
	x0::X
	vrg::VRG
end
function initialize!(alg::SAGA)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::SAGA, iter, stage)
	i = rand(alg.rng, 1:nfunc(alg.vrg))
	addandstore_vrgrad!(alg.x, alg.vrg, i, -alg.stepsize, -alg.stepsize)
end
stageupdate!(alg::SAGA, stage) = 1
primiterates(alg::SAGA) = alg.x
dualiterates(alg::SAGA) = alg.vrg


##############################
# SVRG
struct SVRG{T,I,RNG,X,VRG} <: VRAlgorithm
	stepsize::T
	stagelen::I
	rng::RNG
	x::X
	x0::X
	vrg::VRG
end
function initialize!(alg::SVRG)
	reset!(alg.vrg)
	alg.x .= alg.x0
end
function update!(alg::SVRG, iter, stage)
	i = rand(alg.rng, 1:nfunc(alg.vrg))
	add_vrgrad!(alg.x, alg.vrg, i, -alg.stepsize, -alg.stepsize)
end
function stageupdate!(alg::SVRG, stage)
	store_grad!(alg.vrg, alg.x)
	return alg.stagelen
end
primiterates(alg::SVRG) = alg.x
dualiterates(alg::SVRG) = alg.vrg
