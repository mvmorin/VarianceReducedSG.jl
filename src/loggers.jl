export ShowIterations,
	ShowNewLine,
	ShowTime,
	ShowFuncVal,
	NoLog,
	StoreFuncVal

export initialize!,
	finalize!,
	stagelog!,
	log!

################################################################################
# Loggers
################################################################################
abstract type AbstractLogger end

initialize!(l::AbstractLogger) = nothing
finalize!(l::AbstractLogger) = nothing

stagelog!(l::AbstractLogger) = nothing
stagelog!(l::AbstractLogger, stage) = stagelog!(l::AbstractLogger)
stagelog!(l::AbstractLogger, iter, stage) = stagelog!(l::AbstractLogger, stage)
stagelog!(l::AbstractLogger, alg, iter, stage) =
	stagelog!(l::AbstractLogger, iter, stage)

log!(l::AbstractLogger) = nothing
log!(l::AbstractLogger, iter) = log!(l::AbstractLogger)
log!(l::AbstractLogger, iter, stage) = log!(l::AbstractLogger, iter)
log!(l::AbstractLogger, alg, iter, stage) = log!(l::AbstractLogger, iter, stage)



# Make it possible to package multiple loggers in a tuple.
Loggers = NTuple{N,AbstractLogger} where N

initialize!(tl::Loggers) = unrolled_foreach(initialize!, tl)
finalize!(tl::Loggers) = unrolled_foreach(finalize!, tl)

stagelog!(tl::Loggers, alg, iter, stage) =
	unrolled_foreach(l -> stagelog!(l, alg, stage), tl)

log!(tl::Loggers, alg, iter, stage) =
	unrolled_foreach(l -> log!(l, alg, iter, stage), tl)



################################################################################
struct NoLog <: AbstractLogger end

################################################################################
struct ShowIterations <: AbstractLogger end
log!(l::ShowIterations, iter, stage) =
	@printf("Stage: %5d, Iteration: %7d, ", stage, iter)


################################################################################
struct ShowNewLine{S} <: AbstractLogger
	s::S
end
ShowNewLine() = ShowNewLine("")
log!(l::ShowNewLine) = println(l.s)


################################################################################
struct ShowTime{F} <: AbstractLogger
	t0::Vector{F}
	function ShowTime()
		t0 = time()
		new{typeof(t0)}([t0])
	end
end
initialize!(l::ShowTime) = l.t0[1] = time()
finalize!(l::ShowTime) = @printf("Total time elapsed: %.5g\n", time() - l.t0[1])
log!(l::ShowTime) = @printf("Time: %5.f, ", time() - l.t0[1])


################################################################################
struct ShowFuncVal{F,L} <: AbstractLogger
	f::F
	label::L
end
log!(l::ShowFuncVal, alg, iter, stage) = 
	@printf("%s: %g, ", l.label, l.f(primiterates(alg), dualiterates(alg)))


################################################################################
struct StoreFuncVal{F,V,VI<:AbstractArray{Int}} <: AbstractLogger
	f::F
	fvals::V
	iterations::VI
	stages::VI
	idx::Vector{Int}
	StoreFuncVal(f::F, fv::V, i::VI, s::VI) where {F,V,VI} =
		new{F,V,VI}(f,fv,i,s,[1])
end
initialize!(l::StoreFuncVal) = l.idx[1] = 1
function log!(l::StoreFuncVal, alg, iter, stage)
	i = l.idx[1]

	i > length(l.fvals) && return

	l.fvals[i] = l.f(primiterates(alg), dualiterates(alg))
	l.iterations[i] = iter
	l.stages[i] = stage
	l.idx[1] += 1
end
function finalize!(l::StoreFuncVal)
	# Zero out unused space
	last = l.idx[1]

	fval = zero(eltype(l.fvals))
	iter = 0
	stage = 0
	for i = last:length(l.fvals)
		l.fvals[i] = fval
		l.iterations[i] = iter
		l.stages[i] = stage
	end
end
