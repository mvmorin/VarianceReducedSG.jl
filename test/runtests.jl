module RunTests

println("Testing")

include("gradientstoragetests.jl")
include("leastsquares.jl")
include("lasso.jl")
include("loggertests.jl")

end

