export VRGradient, LinearVRG, UniformVRG

################################################################################
# Variance Reduced Gradient Storage
################################################################################
abstract type AbstractVRGradient end

nfunc(vrg::AbstractVRGradient) = not_implemented()

Base.getindex(vrg::AbstractVRGradient, i::Int) = not_implemented()

reset!(vrg::AbstractVRGradient) = not_implemented()

store_grad!(vrg::AbstractVRGradient, x, ids) = not_implemented()

add_grad!(res, vrg::AbstractVRGradient, x, i, step) = not_implemented()

add_approx!(res, vrg::AbstractVRGradient, step) = not_implemented()

add_vrgrad!(res, vrg::AbstractVRGradient, x, i, step_innv, step_approx) =
	not_implemented()

addandstore_grad!(res, vrg::AbstractVRGradient, x, i, step) = not_implemented()

addandstore_vrgrad!(
		res, vrg::AbstractVRGradient, x, i, step_innv, step_approx) =
	not_implemented()

not_implemented() = error("Not Implemented.")



##############################
# Convenience remappings
# For instance: Calling with same output as input (should always be a valid)

store_grad!(vrg::AbstractVRGradient, x) = store_grad!(vrg, x, 1:nfunc(vrg))

add_grad!(x, vrg::AbstractVRGradient, i, step) =
	add_grad!(x,vrg,x,i,step)

add_vrgrad!(x, vrg::AbstractVRGradient, i, step_innv, step_approx) =
	add_vrgrad!(x,vrg,x,i,step_innv,step_approx)

addandstore_grad!(x, vrg::AbstractVRGradient, i, step) =
	addandstore_grad!(x,vrg,x,i,step)

addandstore_vrgrad!(x, vrg::AbstractVRGradient, i, step_innv, step_approx) =
	addandstore_vrgrad!(x,vrg,x,i,step_innv,step_approx)



################################################################################
struct VRGradient{T<:Real, Vg<:AbstractArray{T}, GF} <: AbstractVRGradient
	grad!::GF
	ysum::Vg
	y::Vector{Vg}	
	buf::Vg
	n::Int

	function VRGradient(grad, buf, n)
		ysum = similar(buf)
		y = [similar(buf) for _ = 1:n]
		vrg = new{eltype(buf), typeof(buf), typeof(grad)}(grad, ysum, y, buf, n)
		reset!(vrg)
		return vrg
	end
end

nfunc(vrg::VRGradient) = vrg.n

Base.getindex(vrg::VRGradient, i::Int) = vrg.y[i]

function reset!(vrg::VRGradient)
	vrg.ysum .= zero(eltype(vrg.ysum))
	for y in vrg.y
		y .= zero(eltype(y))
	end
end

function store_grad!(vrg::VRGradient, x, ids)
	for i in ids
		vrg.ysum .-= vrg.y[i]
		vrg.grad!(vrg.y[i], x, i)
		vrg.ysum .+= vrg.y[i]
	end
end

function add_grad!(res, vrg::VRGradient, x, i, step)
	vrg.grad!(vrg.buf, x, i)
	res .+= step.*vrg.buf
end

function add_approx!(res, vrg::VRGradient, step)
	res .+= step./vrg.n.*vrg.ysum
end

function add_vrgrad!(res, vrg::VRGradient, x, i, step_innv, step_approx)
	add_grad!(res,vrg,x,i,step_innv)
	res .-= step_innv.*vrg.y[i]
	add_approx!(res, vrg, step_approx)
end

function addandstore_grad!(res, vrg::VRGradient, x, i, step)
	vrg.ysum .-= vrg.y[i]
	vrg.grad!(vrg.y[i], x, i)
	vrg.ysum .+= vrg.y[i]
	res .+= step.*vrg.y[i]
end

function addandstore_vrgrad!(res, vrg::VRGradient, x, i, step_innv, step_approx)
	vrg.grad!(vrg.buf, x, i)
	res .+= step_innv.*(vrg.buf .- vrg.y[i]) .+ step_approx./vrg.n.*vrg.ysum
	vrg.ysum .+= vrg.buf .- vrg.y[i]
	vrg.y[i] .= vrg.buf
end




################################################################################
struct LinearVRG{
		T<:Real, Vg<:AbstractArray{T}, D<:AbstractArray{T},
		Vd<:AbstractArray{D}, GF
		} <: AbstractVRGradient
	grad::GF
	ysum::Vg
	y::Vector{T}
	data::Vd
	n::Int
	
	function LinearVRG(grad, data, buf)
		y = Vector{eltype(buf)}(undef, length(data))
		vrg = new{
				eltype(buf),typeof(buf),eltype(data),typeof(data),typeof(grad)
				}(
				grad, buf, y, data, length(data)
				)
		reset!(vrg)
		return vrg
	end
end
nfunc(vrg::LinearVRG) = vrg.n

Base.getindex(vrg::LinearVRG, i::Int) = vrg.data[i]*vrg.y[i]

sgrad(vrg, x, i) = vrg.grad(dot(vrg.data[i],x),i)

function reset!(vrg::LinearVRG)
	vrg.ysum .= zero(eltype(vrg.ysum))
	vrg.y .= zero(eltype(vrg.y))
end

function store_grad!(vrg::LinearVRG, x, ids)
	for i in ids
		innv = sgrad(vrg, x, i) - vrg.y[i]
		vrg.ysum .+= vrg.data[i].*innv
		vrg.y[i] += innv
	end
end

function add_grad!(res, vrg::LinearVRG, x, i, step)
	g = sgrad(vrg, x, i)
	res .+= vrg.data[i].*g.*step
end

function add_approx!(res, vrg::LinearVRG, step)
	res .+= vrg.ysum./vrg.n.*step
end

function add_vrgrad!(res, vrg::LinearVRG, x, i, step_innv, step_approx)
	innv = sgrad(vrg, x, i) - vrg.y[i]
	res .+= vrg.data[i].*innv.*step_innv
	add_approx!(res, vrg, step_approx)
end

function addandstore_grad!(res, vrg::LinearVRG, x, i, step)	
	g = sgrad(vrg, x, i)
	res .+= vrg.data[i].*g.*step
	vrg.ysum .+= vrg.data[i].*(g .- vrg.y[i])
	vrg.y[i] = g
end

function addandstore_vrgrad!(
		res, vrg::LinearVRG, x, i, step_innv, step_approx)
	innv = sgrad(vrg, x, i) - vrg.y[i]
	res .+= vrg.data[i].*innv.*step_innv
	add_approx!(res, vrg, step_approx)
	vrg.ysum .+= vrg.data[i].*innv
	vrg.y[i] += innv
end



################################################################################
struct UniformVRG{
		T<:Real, Vg<:AbstractArray{T}, GF
		} <: AbstractVRGradient
	grad!::GF
	ymean::Vg
	xmem::Vg
	buf::Vg
	n::Int
	function UniformVRG(grad, buf, n)
		vrg = new{eltype(buf), typeof(buf), typeof(grad)}(
			grad, buf, similar(buf), similar(buf), n)
		reset!(vrg)
		return vrg
	end
end

nfunc(vrg::UniformVRG) = vrg.n

Base.getindex(vrg::UniformVRG, i::Int) =
	(vrg.grad!(vrg.buf,vrg.xmem,i); vrg.buf)

function reset!(vrg::UniformVRG)
	vrg.xmem .= zero(eltype(vrg.xmem))
	store_grad!(vrg, vrg.xmem)
end

function store_grad!(vrg::UniformVRG, x)
	vrg.xmem .= x
	vrg.grad!(vrg.ymean, x, 1)
	for i = 2:vrg.n
		vrg.grad!(vrg.buf, x, i)
		vrg.ymean .+= vrg.buf
	end
	vrg.ymean ./= vrg.n
end

function add_grad!(res, vrg::UniformVRG, x, i, step)
	vrg.grad!(vrg.buf, x, i)
	res .+= step.*vrg.buf
end

function add_approx!(res, vrg::UniformVRG, step)
	res .+= vrg.ymean.*step
end

function add_vrgrad!(res, vrg::UniformVRG, x, i, step_innv, step_approx)	
	add_grad!(res, vrg, x, i, step_innv)
	vrg.grad!(vrg.buf, vrg.xmem, i)
	res .-= step_innv.*vrg.buf
	add_approx!(res, vrg, step_approx)
end


invalid() = error("UniformVRG only support uniform store operations.")

store_grad!(vrg::UniformVRG, x, ids) = invalid()
addandstore_grad!(res, vrg::UniformVRG, x, i, step) = invalid()
addandstore_vrgrad!(
		res, vrg::UniformVRG, x, i, step_innv, step_approx) = invalid()
