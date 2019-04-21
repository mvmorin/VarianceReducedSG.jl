module VarianceReducedSG

using LinearAlgebra
using SparseArrays
using Unrolled
using Printf

include("gradientstorage.jl")
include("vralgorithms.jl")
include("loggers.jl")



################################################################################
# Main loop
################################################################################
export iterate!

function iterate!(alg::VRAlgorithm, nstages, logger = NoLog(), iterperlog = 1)
	initialize!(logger)
	initialize!(alg)

	log!(logger, alg, 0, 0)
	
	iter = 0
	for stage = 1:nstages
		stagelen = stageupdate!(alg, iter, stage)
		stagelog!(logger, alg, iter, stage)
		
		for _ = 1:stagelen
			iter += 1
			update!(alg, iter, stage)
			
			iter % iterperlog == 0 && log!(logger, alg, iter, stage)
		end
	end

	finalize!(logger)
end

end # module
