module GradientStorageTests

using Test
using VarianceReducedSG
using LinearAlgebra
using SparseArrays
using Random
Random.seed!(0)

using VarianceReducedSG:
	nfunc,
	reset!,
	grad!,
	approx!,
	vrgrad!,
	grad_store!,
	vrgrad_store!


function allsame(g1, g2)
	nfunc(g1) != nfunc(g2) && return false
	length(g1[1]) != length(g2[1]) && return false
	
	dim = length(g1[1])
	n = nfunc(g1)

	pass = true
	for i = 1:n
		test = isapprox(g1[i], g2[i])
		pass = pass && test
	end
	pass
end

function allsameexcept(g1, g2, ids)
	nfunc(g1) != nfunc(g2) && return false
	length(g1[1]) != length(g2[1]) && return false
	
	dim = length(g1[1])
	n = nfunc(g1)
	
	pass = true
	for i in 1:n
		test = isapprox(g1[i], g2[i])
		if i in ids
			pass = pass && !test
		else
			pass = pass && test
		end
	end
	pass
end

function resetstorereset(g1, g2, n, dim)
	reset!(g1)
	reset!(g2)

	@test allsame(g1, g2)

	x = randn(dim)
	i = rand(1:n)
	grad_store!(g1,x,i)
	@test allsameexcept(g1, g2, i)
	
	ids = rand(1:n, Int(ceil(n/10)))
	grad_store!(g1,x,ids)
	@test allsameexcept(g1, g2, [i;ids])

	grad_store!(g1,x)
	@test allsameexcept(g1, g2, 1:n)

	reset!(g1)
	@test allsame(g1, g2)
end

function nonmodifying_calls(g1, g2, n, dim)
	reset!(g1)
	reset!(g2)

	res = randn(dim)
	x = randn(dim)
	x_ctrl = copy(x)
	
	grad!(res, g1, x, rand(1:n))
	@test allsame(g1, g2)
	@test x == x_ctrl
	
	approx!(res, g1)
	@test allsame(g1, g2)

	vrgrad!(res, g1, x, rand(1:n), randn())
	@test allsame(g1, g2)
	@test x == x_ctrl
	
	nfunc(g1)
	@test allsame(g1, g2)
	
	g1[rand(1:n)]
	@test allsame(g1, g2)
end

function modifying_calls(g1, g2, n, dim)
	reset!(g1)
	reset!(g2)

	x = randn(dim)

	res = copy(x)
	i = rand(1:n)
	mod_ids = [i]

	grad_store!(res, g1, x, i)
	@test allsameexcept(g1, g2, mod_ids)
	@test !isapprox(res, x)


	res = copy(x)
	i = rand(1:n)
	mod_ids = [mod_ids; i]
	
	grad_store!(res, g1, x, i)
	@test allsameexcept(g1, g2, mod_ids)
	@test !isapprox(res, x)


	res = copy(x)
	i = rand(1:n)
	mod_ids = [mod_ids; i]

	vrgrad_store!(res, g1, x, i, randn())
	@test allsameexcept(g1, g2, mod_ids)
	@test !isapprox(res, x)


	res = copy(x)
	i = rand(1:n)
	mod_ids = [mod_ids; i]

	vrgrad_store!(res, g1, x, i, randn())
	@test allsameexcept(g1, g2, mod_ids)
	@test !isapprox(res, x)


	reset!(g1)
	@test allsame(g1, g2)
end

function correct_grad(g, n, dim, f)
	# Scramble memories a little
	grad_store!(g, randn(dim))
	grad_store!(g, randn(dim), rand(1:n, Int(ceil(n/7))))
	
	x1 = randn(dim)
	i1 = rand(1:n)
	ctrl1 = similar(x1)
	f(ctrl1,x1,i1)
	
	x2 = randn(dim)
	i2 = rand(1:n)
	ctrl2 = similar(x2)
	f(ctrl2,x2,i2)


	res = randn(dim)
	grad!(res, g, x1, i1)
	@test isapprox(res, ctrl1)
	
	grad_store!(res, g, x2, i2)
	@test isapprox(res, ctrl2)
	

	res = randn(dim)
	grad_store!(res, g, x1, i1)
	@test isapprox(res, ctrl1)
	
	grad!(res, g, x2, i2)
	@test isapprox(res, ctrl2)
		
	
	x = randn(dim)
	i = rand(1:n)
	
	res1 = randn(dim)
	res2 = copy(res1)
	grad!(res1, g, x, i)
	grad!(res2, g, x, i)
	@test isapprox(res1, res2)
	
	res1 = randn(dim)
	res2 = copy(res1)
	grad!(res1, g, x, i)
	grad_store!(res2, g, x, i)
	@test isapprox(res1, res2)
end

function correct_approximation(g, n, dim, f)
	# Uniform storage	
	x = randn(dim)
	
	ctrl = zeros(dim)
	buf = similar(x)
	for i = 1:n
		f(buf, x, i)
		ctrl .+= buf./n
	end
	
	
	grad_store!(g, x)

	x_res = randn(dim)
	approx!(x_res, g)
	@test isapprox(x_res, ctrl)

	vrgrad!(x_res, g, x, rand(1:n), randn())
	@test isapprox(x_res, ctrl)
	
	vrgrad_store!(x_res, g, x, rand(1:n), randn())
	@test isapprox(x_res, ctrl)
	
	approx!(x_res, g)
	@test isapprox(x_res, ctrl)
	
	vrgrad!(x_res, g, x, rand(1:n), randn())
	@test isapprox(x_res, ctrl)

	
	# Non-uniform storage
	old_approx = randn(dim)
	approx!(old_approx, g)

	x = randn(dim)
	i = rand(1:n)
	
	old_y = copy(g[i])
	new_y = zeros(dim)
	f(new_y, x, i)
	
	grad_store!(g, x, i)
	
	new_approx = randn(dim)
	approx!(new_approx, g)
	@test isapprox(new_approx, old_approx + (new_y - old_y)/n)

	
	old_approx = new_approx
	
	x = randn(dim)
	i = rand(1:n)
	old_y = copy(g[i])
	new_y = zeros(dim)
	f(new_y, x, i)
	
	vrgrad_store!(zeros(dim), g, x, i, randn())
	
	new_approx = randn(dim)
	vrgrad!(new_approx, g, randn(dim), rand(1:n), 0)
	@test isapprox(new_approx, old_approx + (new_y - old_y)/n)
	
	
	old_approx = new_approx
	
	x = randn(dim)
	i = rand(1:n)
	old_y = copy(g[i])
	new_y = zeros(dim)
	f(new_y, x, i)
	
	grad_store!(zeros(dim), g, x, i)
	
	new_approx = randn(dim)
	vrgrad_store!(new_approx, g, randn(dim), rand(1:n), 0)
	@test isapprox(new_approx, old_approx + (new_y - old_y)/n)
end

function correct_innovation(g, n, dim, f)
	x = randn(dim)
	grad_store!(g, x)
	
	res = randn(dim)
	res_app = randn(dim)
	approx!(res_app, g)
	vrgrad!(res, g, x, rand(1:n), randn())
	@test isapprox(res, res_app)

	res = randn(dim)
	res_app = randn(dim)
	approx!(res_app, g)
	vrgrad_store!(res, g, x, rand(1:n), randn())
	@test isapprox(res, res_app)
	
	res = randn(dim)
	res_app = randn(dim)
	approx!(res_app, g)
	vrgrad!(res, g, x, rand(1:n), randn())
	@test isapprox(res, res_app)


	x = randn(dim)
	i = rand(1:n)
	old_y = copy(g[i])
	new_y = zeros(dim)
	f(new_y, x, i)

	step = randn()
	res = zeros(dim)
	res_app = randn(dim)
	approx!(res_app, g)
	vrgrad!(res, g, x, i, step)
	@test isapprox(res, res_app + step*(new_y - old_y))
	
	step = randn()
	res = zeros(dim)
	res_app = randn(dim)
	approx!(res_app, g)
	vrgrad_store!(res, g, x, i, step)
	@test isapprox(res, res_app + step*(new_y - old_y))
	
	step = randn()
	res = zeros(dim)
	res_app = randn(dim)
	approx!(res_app, g)
	vrgrad!(res, g, x, i, step)
	@test isapprox(res, res_app)


	x = randn(dim)
	i = rand(1:n)
	step_innv = randn()
	
	vr_ctrl = zeros(dim)
	tmp = zeros(dim)
	vr_ctrl .-= step_innv.*g[i]
	grad!(tmp, g, x, i)
	vr_ctrl .+= step_innv.*tmp
	approx!(tmp, g)
	vr_ctrl .+= tmp

	vr = randn(dim)
	vrgrad!(vr, g, x, i, step_innv)
	@test isapprox(vr, vr_ctrl)

	res = randn(dim)
	vrgrad!(res, g, x, i, step_innv)
	@test isapprox(res, vr_ctrl)
	
	res = copy(x)
	vrgrad_store!(res, g, x, i, step_innv)
	@test isapprox(res, vr_ctrl)
end


function equivalent(g1, g2, n, dim; uniform_store = false)
	x = randn(dim)
	grad_store!(g1, x)
	grad_store!(g2, x)
	@test allsame(g1,g2)

	if !uniform_store
		x = randn(dim)
		ids = 1:3:n
		grad_store!(g1, x, ids)
		grad_store!(g2, x, ids)
		@test allsame(g1,g2)


		x = randn(dim)
		i = rand(1:n)

		res1 = randn(dim)
		res2 = randn(dim)
		grad_store!(res1, g1, x, i)
		grad_store!(res2, g2, x, i)
		@test isapprox(res1, res2)


		x = randn(dim)
		i = rand(1:n)
		innv = randn()

		res1 = randn(dim)
		res2 = randn(dim)
		x_ctrl = copy(x)
		vrgrad_store!(res1, g1, x, i, innv)
		vrgrad_store!(res2, g2, x, i, innv)
		@test isapprox(res1, res2)
		@test isapprox(x, x_ctrl)
	end
	
	x = randn(dim)
	i = rand(1:n)
	
	res1 = randn(dim)
	res2 = randn(dim)
	grad!(res1, g1, x, i)
	grad!(res2, g2, x, i)
	@test isapprox(res1, res2)
	

	res1 = randn(dim)
	res2 = randn(dim)
	approx!(res1, g1)
	approx!(res2, g2)
	@test isapprox(res1, res2)
	
	
	x = randn(dim)
	i = rand(1:n)
	innv = randn()

	res1 = randn(dim)
	res2 = randn(dim)
	x_ctrl = copy(x)
	vrgrad!(res1, g1, x, i, innv)
	vrgrad!(res2, g2, x, i, innv)
	@test isapprox(res1, res2)
	@test isapprox(x, x_ctrl)
end

function allocation(g,n,dim; uniform_store=false)
	nfunc(g) #run once to make sure it is compiled
	@test 0 == @allocated nfunc(g)
	#g[1]
	#@test 0 == @allocated g[1]
	reset!(g)
	@test 0 == @allocated reset!(g)

	x = randn(dim)
	res = randn(dim)
	ids = 1:3:n
	i = rand(1:n)
	step_innv = randn()

	if uniform_store
		grad_store!(g,x)
		@test 0 == @allocated grad_store!(g,x)
	else
		grad_store!(g,x,ids)
		@test 0 == @allocated grad_store!(g,x,ids)

		grad_store!(res, g, x, i)
		@test 0 == @allocated grad_store!(res, g, x, i)

		vrgrad_store!(res, g, x, i ,step_innv)
		@test 0 == @allocated vrgrad_store!(res, g, x, i, step_innv)
	end


	grad!(res, g, x, i)
	@test 0 == @allocated grad!(res, g, x, i)

	approx!(res, g)
	@test 0 == @allocated approx!(res, g)

	vrgrad!(res, g, x, i, step_innv)
	@test 0 == @allocated vrgrad!(res, g, x, i, step_innv)
end



@testset "VRGradient" begin
	isq(y,x,i) = (y .= i.*x.^2)
	
	for dim in [1,101], n in [1,313]
		g1 = VRGradient(isq, zeros(dim), n)
		g2 = VRGradient(isq, zeros(dim), n)

		@test allsame(g1,g2)
		@test nfunc(g1) == n
		@test nfunc(g2) == n

		resetstorereset(g1, g2, n, dim)

		nonmodifying_calls(g1, g2, n, dim)
		modifying_calls(g1, g2, n, dim)

		correct_grad(g1, n, dim, isq)
		correct_approximation(g1, n, dim, isq)
		correct_innovation(g1, n, dim, isq)

		allocation(g1, n, dim)
	end
end

@testset "LinearVRG" begin
	sq(x, i) = i*x^2
	
	for dim in [1,101], n in [1,313]
		data = [randn(dim) for _ = 1:n]
		nzero = Int(ceil(dim/17))
		for d in data
			d[rand(1:dim, dim - nzero)] .= 0.0
		end

		isq(y,x,i) = (y .= data[i]*sq(dot(data[i],x), i))

		g1 = LinearVRG(sq, data, zeros(dim))
		g2 = LinearVRG(sq, data, zeros(dim))

		@test allsame(g1,g2)
		@test nfunc(g1) == n
		@test nfunc(g2) == n

		resetstorereset(g1, g2, n, dim)

		nonmodifying_calls(g1, g2, n, dim)
		modifying_calls(g1, g2, n, dim)

		correct_grad(g1, n, dim, isq)
		correct_approximation(g1, n, dim, isq)
		correct_innovation(g1, n, dim, isq)

		g_ctrl = VRGradient(isq, zeros(dim), n)
		equivalent(g1, g_ctrl, n, dim)

		allocation(g1, n, dim)
	end
end

@testset "LinearVRG - Sparse Data" begin
	sq(x, i) = i*x^2

	for dim in [1,101], n in [1,313]
		spdata = [(d = sprandn(dim, 0.5); d[1] = randn(); d) for _ = 1:n]
		data = [ Array(spd) for spd in spdata]

		isq(y,x,i) = (y .= data[i]*sq(dot(data[i],x), i))

		g1 = LinearVRG(sq, spdata, zeros(dim))
		g2 = LinearVRG(sq, spdata, zeros(dim))

		@test allsame(g1,g2)
		@test nfunc(g1) == n
		@test nfunc(g2) == n

		resetstorereset(g1, g2, n, dim)

		nonmodifying_calls(g1, g2, n, dim)
		modifying_calls(g1, g2, n, dim)

		correct_grad(g1, n, dim, isq)
		correct_approximation(g1, n, dim, isq)
		correct_innovation(g1, n, dim, isq)

		gd = LinearVRG(sq, data, zeros(dim))
		equivalent(g1, gd, n, dim)

		allocation(g1, n, dim)
	end
end

@testset "UniformVRG" begin
	isq(y,x,i) = (y .= i.*x.^2)
	
	for dim in [1,101], n in [1,313]
		g1 = UniformVRG(isq, zeros(dim), n)
		g2 = UniformVRG(isq, zeros(dim), n)

		@test allsame(g1,g2)
		@test nfunc(g1) == n
		@test nfunc(g2) == n
		

		x = randn(dim)
		grad_store!(g1,x)
		grad_store!(g2,x)
		@test allsame(g1,g2)
		
		reset!(g1)
		reset!(g2)
		@test allsame(g1,g2)

		nonmodifying_calls(g1, g2, n, dim)
		
		g_ctrl = VRGradient(isq, zeros(dim), n)
		equivalent(g1, g_ctrl, n, dim, uniform_store = true)

		allocation(g1, n, dim, uniform_store = true)

		x = randn(dim)
		i = rand(1:n)
		res = randn(dim)
		innv = randn()
		@test_throws ErrorException grad_store!(g1, x, i)
		@test_throws ErrorException grad_store!(res, g1, x, i)
		@test_throws ErrorException vrgrad_store!(res, g1, x, i, innv)
	end
end

end # module
