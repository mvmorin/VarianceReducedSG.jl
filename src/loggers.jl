export ShowIterations, ShowNewline, ShowTime, ShowFuncVal, NoLog

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
struct ShowNewline{S} <: AbstractLogger
	s::S
end
ShowNewline() = ShowNewline("")
log!(l::ShowNewline) = println(l.s)


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

