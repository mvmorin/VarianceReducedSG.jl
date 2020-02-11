export ShowIterations,
	ShowNewLine,
	ShowTime,
	NoLog,
	CacheFuncVal,
	StoreFuncVal,
	ShowFuncVal,
	StoreLogIterations

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

log!(l::AbstractLogger) = nothing
log!(l::AbstractLogger, iter) = log!(l::AbstractLogger)
log!(l::AbstractLogger, iter, stage) = log!(l::AbstractLogger, iter)
log!(l::AbstractLogger, alg, iter, stage) = log!(l::AbstractLogger, iter, stage)


# Make it possible to package multiple loggers in a tuple.
Loggers = NTuple{N,AbstractLogger} where N

initialize!(tl::Loggers) = unrolled_foreach(initialize!, tl)
finalize!(tl::Loggers) = unrolled_foreach(finalize!, tl)

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
	t0::Base.RefValue{F}
	function ShowTime()
		t0 = time()
		new{typeof(t0)}(Ref(t0))
	end
end
initialize!(l::ShowTime) = l.t0[] = time()
finalize!(l::ShowTime) = @printf("Total time elapsed: %.5g\n", time() - l.t0[])
log!(l::ShowTime) = @printf("Time: %5.f, ", time() - l.t0[])


################################################################################
struct CacheFuncVal{F,T} <: AbstractLogger
	f::F
	val::Base.RefValue{T}
	CacheFuncVal(f::F,T=Float64) where {F} = new{F,T}(f,Ref(zero(T)))
end
log!(l::CacheFuncVal,alg,iter,stage) =
	l.val[] = l.f(primiterates(alg),dualiterates(alg))
(l::CacheFuncVal)(x,y) = l.val[]


################################################################################
struct ShowFuncVal{F,L} <: AbstractLogger
	f::F
	label::L
end
log!(l::ShowFuncVal, alg, iter, stage) = 
	@printf("%s: %g, ", l.label, l.f(primiterates(alg), dualiterates(alg)))


################################################################################
struct StoreFuncVal{F,V} <: AbstractLogger
	f::F
	fvals::V
	idx::Base.RefValue{Int}
	StoreFuncVal(f::F, fv::V) where {F,V} = new{F,V}(f,fv,Ref(1))
end
initialize!(l::StoreFuncVal) = l.idx[] = 1
function log!(l::StoreFuncVal, alg, iter, stage)
	i = l.idx[]
	i > length(l.fvals) && return
	l.fvals[i] = l.f(primiterates(alg), dualiterates(alg))
	l.idx[] += 1
end
function finalize!(l::StoreFuncVal)
	# Zero out unused space
	last = l.idx[]

	fval = zero(eltype(l.fvals))
	for i = last:length(l.fvals)
		l.fvals[i] = fval
	end
end

################################################################################
struct StoreLogIterations{VI<:AbstractArray{Int}} <: AbstractLogger
	iterations::VI
	idx::Base.RefValue{Int}
	StoreLogIterations(iterations::VI) where {VI} = new{VI}(iterations,Ref(1))
end
initialize!(l::StoreLogIterations) = l.idx[] = 1
function log!(l::StoreLogIterations, alg, iter, stage)
	i = l.idx[]
	i > length(l.iterations) && return
	l.iterations[i] = iter
	l.idx[] += 1
end
function finalize!(l::StoreLogIterations)
	# Zero out unused space
	last = l.idx[]

	iter = zero(eltype(l.iterations))
	for i = last:length(l.iterations)
		l.iterations[i] = iter
	end
end
