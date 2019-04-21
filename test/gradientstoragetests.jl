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
	add_grad!,
	add_approx!,
	add_vrgrad!,
	addandstore_grad!,
	addandstore_vrgrad!,
	store_grad!


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
	store_grad!(g1,x,i)
	@test allsameexcept(g1, g2, i)
	
	ids = rand(1:n, Int(ceil(n/10)))
	store_grad!(g1,x,ids)
	@test allsameexcept(g1, g2, [i;ids])

	store_grad!(g1,x)
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
	
	add_grad!(res, g1, x, rand(1:n), randn())
	@test allsame(g1, g2)
	@test x == x_ctrl
	
	add_approx!(res, g1, randn())
	@test allsame(g1, g2)

	add_vrgrad!(res, g1, x, rand(1:n), randn(), randn())
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

	addandstore_grad!(res, g1, x, i, randn())
	@test allsameexcept(g1, g2, mod_ids)
	@test !isapprox(res, x)


	res = copy(x)
	i = rand(1:n)
	mod_ids = [mod_ids; i]
	
	addandstore_grad!(res, g1, x, i, randn())
	@test allsameexcept(g1, g2, mod_ids)
	@test !isapprox(res, x)


	res = copy(x)
	i = rand(1:n)
	mod_ids = [mod_ids; i]

	addandstore_vrgrad!(res, g1, x, i, randn(), randn())
	@test allsameexcept(g1, g2, mod_ids)
	@test !isapprox(res, x)


	res = copy(x)
	i = rand(1:n)
	mod_ids = [mod_ids; i]

	addandstore_vrgrad!(res, g1, x, i, randn(), randn())
	@test allsameexcept(g1, g2, mod_ids)
	@test !isapprox(res, x)


	reset!(g1)
	@test allsame(g1, g2)
end

function correct_grad(g, n, dim, f)
	# Scramble memories a little
	store_grad!(g, randn(dim))
	store_grad!(g, randn(dim), rand(1:n, Int(ceil(n/7))))
	
	x1 = randn(dim)
	i1 = rand(1:n)
	ctrl1 = similar(x1)
	f(ctrl1,x1,i1)
	
	x2 = randn(dim)
	i2 = rand(1:n)
	ctrl2 = similar(x2)
	f(ctrl2,x2,i2)


	res = zeros(dim)
	step1 = randn()
	step2 = randn()
	
	add_grad!(res, g, x1, i1, step1)
	@test isapprox(res, step1*ctrl1)
	
	addandstore_grad!(res, g, x2, i2, step2)
	@test isapprox(res, step1*ctrl1 + step2*ctrl2)
	

	res = zeros(dim)
	step1 = randn()
	step2 = randn()
	
	addandstore_grad!(res, g, x1, i1, step1)
	@test isapprox(res, step1*ctrl1)
	
	add_grad!(res, g, x2, i2, step2)
	@test isapprox(res, step1*ctrl1 + step2*ctrl2)
		
	
	x = randn(dim)
	i = rand(1:n)
	step = randn()
	
	res1 = randn(dim)
	res2 = copy(res1)
	add_grad!(res1, g, x, i, step)
	add_grad!(res2, g, x, i, step)
	@test isapprox(res1, res2)
	
	res1 = randn(dim)
	res2 = copy(res1)
	add_grad!(res1, g, x, i, step)
	addandstore_grad!(res2, g, x, i, step)
	@test isapprox(res1, res2)
end

function correct_approximation(g, n, dim, f)
	# Uniform storage	
	x = randn(dim)
	step = randn()
	
	ctrl = copy(x)
	buf = similar(x)
	for i = 1:n
		f(buf, x, i)
		ctrl .+= step.*buf./n
	end
	
	
	store_grad!(g, x)

	x_res = copy(x)
	add_approx!(x_res, g, step)
	@test isapprox(x_res, ctrl)

	x_res = copy(x)
	add_vrgrad!(x_res, g, x, rand(1:n), randn(), step)
	@test isapprox(x_res, ctrl)
	
	x_res = copy(x)
	addandstore_vrgrad!(x_res, g, x, rand(1:n), randn(), step)
	@test isapprox(x_res, ctrl)
	
	x_res = copy(x)
	add_approx!(x_res, g, step)
	@test isapprox(x_res, ctrl)
	
	x_res = copy(x)
	add_vrgrad!(x_res, g, x, rand(1:n), randn(), step)
	@test isapprox(x_res, ctrl)

	
	# Non-uniform storage
	old_approx = zeros(dim)
	add_approx!(old_approx, g, 1.0)

	x = randn(dim)
	i = rand(1:n)
	
	old_y = copy(g[i])
	new_y = zeros(dim)
	f(new_y, x, i)
	
	store_grad!(g, x, i)
	
	new_approx = zeros(dim)
	add_approx!(new_approx, g, 1.0)
	@test isapprox(new_approx, old_approx + (new_y - old_y)/n)

	
	old_approx = new_approx
	
	x = randn(dim)
	i = rand(1:n)
	old_y = copy(g[i])
	new_y = zeros(dim)
	f(new_y, x, i)
	
	addandstore_vrgrad!(zeros(dim), g, x, i, randn(), randn())
	
	new_approx = zeros(dim)
	add_vrgrad!(new_approx, g, randn(dim), rand(1:n), 0, 1)
	@test isapprox(new_approx, old_approx + (new_y - old_y)/n)
	
	
	old_approx = new_approx
	
	x = randn(dim)
	i = rand(1:n)
	old_y = copy(g[i])
	new_y = zeros(dim)
	f(new_y, x, i)
	
	addandstore_grad!(zeros(dim), g, x, i, randn())
	
	new_approx = zeros(dim)
	addandstore_vrgrad!(new_approx, g, randn(dim), rand(1:n), 0, 1)
	@test isapprox(new_approx, old_approx + (new_y - old_y)/n)
end

function correct_innovation(g, n, dim, f)
	x = randn(dim)
	store_grad!(g, x)
	
	res = zeros(dim)
	add_vrgrad!(res, g, x, rand(1:n), randn(), 0.0)
	@test isapprox(res, zeros(dim), atol=1e-10)

	res = zeros(dim)
	addandstore_vrgrad!(res, g, x, rand(1:n), randn(), 0.0)
	@test isapprox(res, zeros(dim), atol=1e-10)
	
	res = zeros(dim)
	add_vrgrad!(res, g, x, rand(1:n), randn(), 0.0)
	@test isapprox(res, zeros(dim), atol=1e-10)


	x = randn(dim)
	i = rand(1:n)
	old_y = copy(g[i])
	new_y = zeros(dim)
	f(new_y, x, i)

	step = randn()
	res = zeros(dim)
	add_vrgrad!(res, g, x, i, step, 0.0)
	@test isapprox(res, step*(new_y - old_y))
	
	step = randn()
	res = zeros(dim)
	addandstore_vrgrad!(res, g, x, i, step, 0.0)
	@test isapprox(res, step*(new_y - old_y))
	
	step = randn()
	res = zeros(dim)
	add_vrgrad!(res, g, x, i, step, 0.0)
	@test isapprox(res, zeros(dim), atol=1e-10)


	x = randn(dim)
	i = rand(1:n)
	step_innv = randn()
	step_approx = randn()
	
	vr_ctrl = zeros(dim)
	vr_ctrl .-= step_innv.*g[i]
	add_grad!(vr_ctrl, g, x, i, step_innv)
	add_approx!(vr_ctrl, g, step_approx)

	vr = zeros(dim)
	add_vrgrad!(vr, g, x, i, step_innv, step_approx)
	@test isapprox(vr, vr_ctrl)

	res = copy(x)
	add_vrgrad!(res, g, x, i, step_innv, step_approx)
	@test isapprox(res, x + vr_ctrl)
	
	res = copy(x)
	addandstore_vrgrad!(res, g, x, i, step_innv, step_approx)
	@test isapprox(res, x + vr_ctrl)
	
	
	x = randn(dim)
	i = rand(1:n)
	step_innv = randn()
	step_approx = randn()
	
	res1 = randn(dim)
	res2 = copy(res1)
	add_vrgrad!(res1, g, x, i, step_innv, step_approx)
	add_vrgrad!(res2, g, x, i, step_innv, step_approx)
	@test isapprox(res1, res2)
	
	res1 = randn(dim)
	res2 = copy(res1)
	add_vrgrad!(res1, g, x, i, step_innv, step_approx)
	addandstore_vrgrad!(res2, g, x, i, step_innv, step_approx)
	@test isapprox(res1, res2)
end

function input_overwriting(g1, g2, n, dim)
	reset!(g1)
	reset!(g2)

	# Scramble memories
	x = randn(dim)
	store_grad!(g1, x)
	store_grad!(g2, x)
	for _ = 1:n
		x = randn(dim)
		i = rand(1:n)
		store_grad!(g1, x, i)
		store_grad!(g2, x, i)
	end
	
	x = randn(dim)
	i = rand(1:n)
	step_innv = randn()
	step_approx = randn()
	
	res1 = copy(x)
	res2 = copy(x)
	addandstore_vrgrad!(res1, g1, i, step_innv, step_approx)
	addandstore_vrgrad!(res2, g2, x, i, step_innv, step_approx)
	@test isapprox(res1, res2)
	
	
	x = randn(dim)
	i = rand(1:n)
	step_innv = randn()
	step_approx = randn()
	
	res1 = copy(x)
	res2 = copy(x)
	add_vrgrad!(res1, g1, i, step_innv, step_approx)
	add_vrgrad!(res2, g2, x, i, step_innv, step_approx)
	@test isapprox(res1, res2)
end

function equivalent(g1, g2, n, dim; uniform_store = false)
	x = randn(dim)
	store_grad!(g1, x)
	store_grad!(g2, x)
	@test allsame(g1,g2)

	if !uniform_store
		x = randn(dim)
		ids = 1:3:n
		store_grad!(g1, x, ids)
		store_grad!(g2, x, ids)
		@test allsame(g1,g2)


		x = randn(dim)
		i = rand(1:n)
		step = randn()

		res1 = randn(dim)
		res2 = copy(res1)
		addandstore_grad!(res1, g1, x, i, step)
		addandstore_grad!(res2, g2, x, i, step)
		@test isapprox(res1, res2)


		x = randn(dim)
		i = rand(1:n)
		step_innv = randn()
		step_approx = randn()

		res1 = randn(dim)
		res2 = copy(res1)
		x_ctrl = copy(x)
		addandstore_vrgrad!(res1, g1, x, i, step_innv, step_approx)
		addandstore_vrgrad!(res2, g2, x, i, step_innv, step_approx)
		@test isapprox(res1, res2)
		@test isapprox(x, x_ctrl)
	end

	
	x = randn(dim)
	i = rand(1:n)
	step = randn()
	
	res1 = randn(dim)
	res2 = copy(res1)
	add_grad!(res1, g1, x, i, step)
	add_grad!(res2, g2, x, i, step)
	@test isapprox(res1, res2)
	

	step = randn()
	res1 = randn(dim)
	res2 = copy(res1)
	add_approx!(res1, g1, step)
	add_approx!(res2, g2, step)
	@test isapprox(res1, res2)
	
	
	x = randn(dim)
	i = rand(1:n)
	step_innv = randn()
	step_approx = randn()

	res1 = randn(dim)
	res2 = copy(res1)
	x_ctrl = copy(x)
	add_vrgrad!(res1, g1, x, i, step_innv, step_approx)
	add_vrgrad!(res2, g2, x, i, step_innv, step_approx)
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
	step_approx = randn()

	if uniform_store
		store_grad!(g,x)
		@test 0 == @allocated store_grad!(g,x)
	else
		store_grad!(g,x,ids)
		@test 0 == @allocated store_grad!(g,x,ids)

		addandstore_grad!(res, g, x, i , step_innv)
		@test 0 == @allocated addandstore_grad!(res, g, x, i , step_innv)

		addandstore_vrgrad!(res, g, x, i , step_innv, step_approx)
		@test 0 == @allocated addandstore_vrgrad!(
								res, g, x, i , step_innv, step_approx)
	end


	add_grad!(res, g, x, i, step_innv)
	@test 0 == @allocated add_grad!(res, g, x, i, step_innv)

	add_approx!(res, g, step_approx)
	@test 0 == @allocated add_approx!(res, g, step_approx)

	add_vrgrad!(res, g, x, i, step_innv, step_approx)
	@test 0 == @allocated add_vrgrad!(res, g, x, i, step_innv, step_approx)
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

		input_overwriting(g1, g2, n, dim)

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

		input_overwriting(g1, g2, n, dim)

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

		input_overwriting(g1, g2, n, dim)

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
		store_grad!(g1,x)
		store_grad!(g2,x)
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
		s1 = randn()
		s2 = randn()
		@test_throws ErrorException store_grad!(g1, x, i)
		@test_throws ErrorException addandstore_grad!(res, g1, x, i, s1)
		@test_throws ErrorException addandstore_vrgrad!(res, g1, x, i, s1, s2)
	end
end

end # module
