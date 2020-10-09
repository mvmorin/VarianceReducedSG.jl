export VRGradient, LinearVRG, UniformVRG

################################################################################
# Variance Reduced Gradient Storage
################################################################################
abstract type AbstractVRGradient{N} end

Base.length(vrg::AbstractVRGradient{N}) where N = N

Base.firstindex(vrg::AbstractVRGradient) = 1
Base.lastindex(vrg::AbstractVRGradient) = length(vrg)
Base.iterate(vrg::AbstractVRGradient) = (i = firstindex(vrg); iterate(vrg,i))
Base.iterate(vrg::AbstractVRGradient, i) =
	i > lastindex(vrg) ? nothing : (vrg[i], i+1)


Base.getindex(vrg::AbstractVRGradient, i::Int) = not_implemented()

reset!(vrg::AbstractVRGradient) = not_implemented()

grad!(res, vrg::AbstractVRGradient, x, i) = not_implemented()
vrgrad!(res, vrg::AbstractVRGradient, x, i, innv_weight) = not_implemented()
innov!(res, vrg::AbstractVRGradient, x, i) = not_implemented()
grad_approx!(res, vrg::AbstractVRGradient) = not_implemented()

store_from_point!(vrg::AbstractVRGradient, x, ids) = not_implemented()
store_from_grad!(vrg::AbstractVRGradient, grad, state) = not_implemented()
store_from_innov!(vrg::AbstractVRGradient, innov, state) = not_implemented()

vrgrad_store!(res, vrg::AbstractVRGradient, x, i, innv_weight)=not_implemented()


not_implemented() = error("Not Implemented.")


# Convenience remappings
store_from_point!(vrg::AbstractVRGradient, x) =
	store_from_point!(vrg,x,1:length(vrg))

function grad!(res, vrg::AbstractVRGradient, x)
	grad!(res, vrg, x, 1)
	res_i = similar(res)
	for i = 2:length(vrg)
		grad!(res_i, vrg, x, i)
		res .+= res_i
	end
	res ./= length(vrg)
	return res
end



################################################################################
struct VRGradient{T<:Real, Vg<:AbstractArray{T}, GF, N} <: AbstractVRGradient{N}
	grad!::GF
	ysum::Vg
	y::Vector{Vg}

	function VRGradient(grad!, buf, n)
		ysum = similar(buf)
		y = [similar(buf) for _ = 1:n]
		vrg = new{eltype(buf), typeof(buf), typeof(grad!), n}(grad!, ysum, y)
		reset!(vrg)
		return vrg
	end
end

Base.getindex(vrg::VRGradient, i::Int) = vrg.y[i]

function reset!(vrg::VRGradient)
	vrg.ysum .= zero(eltype(vrg.ysum))
	for y in vrg.y
		y .= zero(eltype(y))
	end
end

function grad!(res, vrg::VRGradient, x, i)
	vrg.grad!(res, x, i)
	return i
end

function vrgrad!(res, vrg::VRGradient, x, i, innv_weight)
	grad!(res,vrg,x,i)
	res .-= vrg[i]
	res .*= innv_weight
	res .+= vrg.ysum./length(vrg)
end

function innov!(res, vrg::VRGradient, x, i)
	grad!(res,vrg,x,i)
	res .-= vrg[i]
	return i
end

grad_approx!(res, vrg::VRGradient) = (res .= vrg.ysum./length(vrg))

function store_from_point!(vrg::VRGradient, x, ids)
	for i in ids
		vrg.ysum .-= vrg[i]
		vrg.grad!(vrg[i], x, i)
		vrg.ysum .+= vrg[i]
	end
end

function store_from_grad!(vrg::VRGradient, grad, state)
	i = state
	vrg.ysum .-= vrg[i]
	vrg[i] .= grad
	vrg.ysum .+= grad
end

function store_from_innov!(vrg::VRGradient, innov, state)
	i = state
	vrg.ysum .+= innov
	vrg[i] .+= innov
end

function vrgrad_store!(res, vrg::VRGradient, x, i, innv_weight)
	grad_approx!(res, vrg)
	res .-= innv_weight.*vrg.y[i]
	vrg.ysum .-= vrg.y[i]
	grad!(vrg.y[i],vrg,x,i)
	res .+= innv_weight.*vrg.y[i]
	vrg.ysum .+= vrg.y[i]
end




################################################################################
struct LinearVRG{
		T<:Real, Vg<:AbstractArray{T}, D<:AbstractArray{T},
		Vd<:AbstractArray{D}, GF, N
		} <: AbstractVRGradient{N}
	grad::GF
	ysum::Vg
	y::Vector{T}
	data::Vd

	function LinearVRG(grad, data, buf)
		y = Vector{eltype(buf)}(undef, length(data))
		vrg = new{
				eltype(buf), typeof(buf), eltype(data),
				typeof(data), typeof(grad), length(data)
				}(
				grad, buf, y, data
				)
		reset!(vrg)
		return vrg
	end
end

Base.getindex(vrg::LinearVRG, i::Int) = vrg.data[i]*vrg.y[i]

function reset!(vrg::LinearVRG)
	vrg.ysum .= zero(eltype(vrg.ysum))
	vrg.y .= zero(eltype(vrg.y))
end

sgrad(vrg, x, i) = vrg.grad(dot(vrg.data[i],x),i)
function grad!(res, vrg::LinearVRG, x, i)
	sg = sgrad(vrg,x,i)
	res .= vrg.data[i].*sg
	return (sg,i)
end

function vrgrad!(res, vrg::LinearVRG, x, i, innv_weight)
	grad_approx!(res, vrg)
	innv = sgrad(vrg, x, i) - vrg.y[i]
	res .+= vrg.data[i].*innv.*innv_weight
end

function innov!(res, vrg::LinearVRG, x, i)
	innv = sgrad(vrg, x, i) - vrg.y[i]
	res .= vrg.data[i].*innv
	return (innv,i)
end

grad_approx!(res, vrg::LinearVRG) = (res .= vrg.ysum./length(vrg))

function store_from_point!(vrg::LinearVRG, x, ids)
	for i in ids
		innv = sgrad(vrg, x, i) - vrg.y[i]
		vrg.ysum .+= vrg.data[i].*innv
		vrg.y[i] += innv
	end
end

function store_from_grad!(vrg::LinearVRG, grad, state)
	(sg,i) = state
	innv = sg - vrg.y[i]
	vrg.ysum .+= vrg.data[i].*innv
	vrg.y[i] += innv
end

function store_from_innov!(vrg::LinearVRG, innov, state)
	(innov_s,i) = state
	vrg.ysum .+= innov
	vrg.y[i] += innov_s
end

function vrgrad_store!(res, vrg::LinearVRG, x, i, innv_weight)
	grad_approx!(res, vrg)
	innv = sgrad(vrg, x, i) - vrg.y[i]
	res .+= vrg.data[i].*innv.*innv_weight
	vrg.ysum .+= vrg.data[i].*innv
	vrg.y[i] += innv
end



################################################################################
struct UniformVRG{
		T<:Real, Vg<:AbstractArray{T}, GF, N
		} <: AbstractVRGradient{N}
	grad!::GF
	ymean::Vg
	xmem::Vg
	buf::Vg
	function UniformVRG(grad, buf, n)
		vrg = new{eltype(buf), typeof(buf), typeof(grad), n}(
			grad, buf, similar(buf), similar(buf))
		reset!(vrg)
		return vrg
	end
end

Base.getindex(vrg::UniformVRG, i::Int) =
	(vrg.grad!(vrg.buf,vrg.xmem,i); vrg.buf)

function reset!(vrg::UniformVRG)
	vrg.xmem .= zero(eltype(vrg.xmem))
	store_from_point!(vrg, vrg.xmem)
end

function grad!(res, vrg::UniformVRG, x, i)
	vrg.grad!(res, x, i)
	return nothing
end

function vrgrad!(res, vrg::UniformVRG, x, i, innv_weight)
	grad_approx!(res, vrg)
	vrg.grad!(vrg.buf, x, i)
	res .+= innv_weight.*vrg.buf
	vrg.grad!(vrg.buf, vrg.xmem, i)
	res .-= innv_weight.*vrg.buf
end

function innov!(res, vrg::UniformVRG, x, i)
	vrg.grad!(res, x, i)
	vrg.grad!(vrg.buf, vrg.xmem, i)
	res .-= vrg.buf
	return nothing
end

grad_approx!(res, vrg::UniformVRG) = (res .= vrg.ymean)

function store_from_point!(vrg::UniformVRG, x)
	vrg.xmem .= x
	vrg.grad!(vrg.ymean, x, 1)
	for i = 2:length(vrg)
		vrg.grad!(vrg.buf, x, i)
		vrg.ymean .+= vrg.buf
	end
	vrg.ymean ./= length(vrg)
end


invalid() = error("UniformVRG only support uniform store operations.")

store_from_point!(vrg::UniformVRG, x, ids) = invalid()
store_from_grad!(vrg::UniformVRG, grad, state) = invalid()
store_from_innov!(vrg::UniformVRG, innov, state) = invalid()
vrgrad_store!(res, vrg::UniformVRG, x, i, innv_weight) = invalid()
