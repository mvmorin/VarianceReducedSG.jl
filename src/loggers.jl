export ShowIterations,
	ShowNewLine,
	ShowTime,
	NoLog,
	CacheFuncVal,
	StoreFuncVal,
	ShowFuncVal,
	StoreLogIterations,
	ShowAlgState,
	StoreAlgState

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
struct ShowAlgState{F,L} <: AbstractLogger
	f::F
	label::L
end
log!(l::ShowAlgState, alg, iter, stage) = print(l.label, ": ", l.f(alg), ", ")


################################################################################
struct LogStorage{V}
	vals::V
	idx::Base.RefValue{Int}
	LogStorage(v::V) where {V} = new{V}(v,Ref(1))
end
initialize!(ls::LogStorage) = ls.idx[] = 1
function store!(ls::LogStorage, val)
	i = ls.idx[]
	i > length(ls.vals) && return
	ls.vals[i] = val
	ls.idx[] += 1
end
function finalize!(ls::LogStorage)
	# Zero out unused space
	last = ls.idx[]
	padding = zero(eltype(ls.vals))
	for i = last:length(ls.vals)
		ls.vals[i] = padding
	end
end


################################################################################
struct StoreFuncVal{F,V} <: AbstractLogger
	f::F
	ls::LogStorage{V}
	StoreFuncVal(f::F, fv::V) where {F,V} = new{F,V}(f,LogStorage(fv))
end
initialize!(l::StoreFuncVal) = initialize!(l.ls)
function log!(l::StoreFuncVal, alg, iter, stage)
	val = l.f(primiterates(alg), dualiterates(alg))
	store!(l.ls, val)
end
finalize!(l::StoreFuncVal) = finalize!(l.ls)


################################################################################
struct StoreLogIterations{VI<:AbstractArray{Int}} <: AbstractLogger
	ls::LogStorage{VI}
	StoreLogIterations(iters::VI) where {VI} = new{VI}(LogStorage(iters))
end
initialize!(l::StoreLogIterations) = initialize!(l.ls)
log!(l::StoreLogIterations, alg, iter, stage) = store!(l.ls, iter)
finalize!(l::StoreLogIterations) = finalize!(l.ls)


################################################################################
struct StoreAlgState{F,V} <: AbstractLogger
	f::F
	ls::LogStorage{V}
	StoreAlgState(f::F, fv::V) where {F,V} = new{F,V}(f,LogStorage(fv))
end
initialize!(l::StoreAlgState) = initialize!(l.ls)
function log!(l::StoreAlgState, alg, iter, stage)
	val = l.f(alg)
	store!(l.ls, val)
end
finalize!(l::StoreAlgState) = finalize!(l.ls)
