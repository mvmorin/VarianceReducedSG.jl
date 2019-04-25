export VRGradient, LinearVRG, UniformVRG

################################################################################
# Variance Reduced Gradient Storage
################################################################################
abstract type AbstractVRGradient{N} end

nfunc(vrg::AbstractVRGradient{N}) where N = N

Base.getindex(vrg::AbstractVRGradient, i::Int) = not_implemented()
reset!(vrg::AbstractVRGradient) = not_implemented()

grad!(res, vrg::AbstractVRGradient, x, i) = not_implemented()
grad_store!(vrg::AbstractVRGradient, x, ids) = not_implemented()
grad_store!(res, vrg::AbstractVRGradient, x, i) = not_implemented()

approx!(res, vrg::AbstractVRGradient) = not_implemented()

vrgrad!(res, vrg::AbstractVRGradient, x, i, innv_weight) = not_implemented()
vrgrad_store!(res, vrg::AbstractVRGradient, x, i, innv_weight)=not_implemented()

not_implemented() = error("Not Implemented.")


# Convenience remappings
grad_store!(vrg::AbstractVRGradient, x) = grad_store!(vrg, x, 1:nfunc(vrg))



################################################################################
struct VRGradient{T<:Real, Vg<:AbstractArray{T}, GF, N} <: AbstractVRGradient{N}
	grad!::GF
	ysum::Vg
	y::Vector{Vg}	
	buf::Vg

	function VRGradient(grad, buf, n)
		ysum = similar(buf)
		y = [similar(buf) for _ = 1:n]
		vrg = new{eltype(buf), typeof(buf), typeof(grad), n}(grad, ysum, y, buf)
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


grad!(res, vrg::VRGradient, x, i) = vrg.grad!(res, x, i)
function grad_store!(vrg::VRGradient, x, ids)
	for i in ids
		vrg.ysum .-= vrg.y[i]
		vrg.grad!(vrg.y[i], x, i)
		vrg.ysum .+= vrg.y[i]
	end
end
function grad_store!(res, vrg::VRGradient, x, i)
	vrg.ysum .-= vrg.y[i]
	vrg.grad!(vrg.y[i], x, i)
	vrg.ysum .+= vrg.y[i]
	res .= vrg.y[i]
end


approx!(res, vrg::VRGradient) = (res .= vrg.ysum./nfunc(vrg))


function vrgrad!(res, vrg::VRGradient, x, i, innv_weight)
	approx!(res, vrg)
	grad!(vrg.buf,vrg,x,i)
	vrg.buf .-= vrg.y[i]
	res .+= innv_weight.*vrg.buf
end
function vrgrad_store!(res, vrg::VRGradient, x, i, innv_weight)
	approx!(res, vrg)
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

sgrad(vrg, x, i) = vrg.grad(dot(vrg.data[i],x),i)

function reset!(vrg::LinearVRG)
	vrg.ysum .= zero(eltype(vrg.ysum))
	vrg.y .= zero(eltype(vrg.y))
end


grad!(res, vrg::LinearVRG, x, i) = (res .= vrg.data[i].*sgrad(vrg,x,i))
function grad_store!(vrg::LinearVRG, x, ids)
	for i in ids
		innv = sgrad(vrg, x, i) - vrg.y[i]
		vrg.ysum .+= vrg.data[i].*innv
		vrg.y[i] += innv
	end
end
function grad_store!(res, vrg::LinearVRG, x, i)
	g = sgrad(vrg, x, i)
	res .= vrg.data[i].*g
	vrg.ysum .+= vrg.data[i].*(g .- vrg.y[i])
	vrg.y[i] = g
end


approx!(res, vrg::LinearVRG) = (res .= vrg.ysum./nfunc(vrg))


function vrgrad!(res, vrg::LinearVRG, x, i, innv_weight)
	approx!(res, vrg)
	innv = sgrad(vrg, x, i) - vrg.y[i]
	res .+= vrg.data[i].*innv.*innv_weight
end
function vrgrad_store!(res, vrg::LinearVRG, x, i, innv_weight)
	approx!(res, vrg)
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
	grad_store!(vrg, vrg.xmem)
end


grad!(res, vrg::UniformVRG, x, i) = vrg.grad!(res, x, i)
function grad_store!(vrg::UniformVRG, x)
	vrg.xmem .= x
	vrg.grad!(vrg.ymean, x, 1)
	for i = 2:nfunc(vrg)
		vrg.grad!(vrg.buf, x, i)
		vrg.ymean .+= vrg.buf
	end
	vrg.ymean ./= nfunc(vrg)
end


approx!(res, vrg::UniformVRG) = (res .= vrg.ymean)


function vrgrad!(res, vrg::UniformVRG, x, i, innv_weight)
	approx!(res, vrg)
	vrg.grad!(vrg.buf, x, i)
	res .+= innv_weight.*vrg.buf
	vrg.grad!(vrg.buf, vrg.xmem, i)
	res .-= innv_weight.*vrg.buf
end


invalid() = error("UniformVRG only support uniform store operations.")

store_grad!(vrg::UniformVRG, x, ids) = invalid()
grad_store!(res, vrg::UniformVRG, x, i) = invalid()
vrgrad_store!(res, vrg::UniformVRG, x, i, innv_weight) = invalid()
