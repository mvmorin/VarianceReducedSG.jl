module VarianceReducedSG

using LinearAlgebra
using ProximalOperators
using SparseArrays
using Unrolled
using Printf
using StatsBase

include("gradientstorage.jl")
include("vralgorithms.jl")
include("loggers.jl")



################################################################################
# Main loop
################################################################################
export iterate!

function iterate!(
		alg::VRAlgorithm, niter, logger=NoLog(), nlogs=1; warmstart=false)

	iterperlog = Int(ceil(niter/nlogs))
	nstages = Int(ceil(max(1,nlogs-1)*iterperlog/expectedstagelen(alg)))

	initialize!(logger)
	!warmstart && initialize!(alg)

	log!(logger, alg, 0, 0)

	iter = 0
	for stage = 1:nstages
		stagelen = stageupdate!(alg, iter, stage)

		for _ = 1:stagelen
			iter += 1
			update!(alg, iter, stage)

			iter % iterperlog == 0 && log!(logger, alg, iter, stage)
		end
	end

	finalize!(logger)
end

end # module
