module ExtremelyRandomizedTrees

using FunctionalDataUtils, ProgressMeter, Compat

export ExtraTrees, predict, predict!

immutable ExtremelyRandomizedTree{T}
    indmatrix::Array{Int,2}
    thresholds::Array{T}
    leafs
    regression
end

z(a...) = zeros(Float32,a...)
immutable Buffers
    px::Array{Float32}
    py::Array{Float32}
    pxy::Array{Float32,2}
    hc::Array{Float32}
    hs::Array{Float32}
    Is::Array{Float32,2}
    Buffers(n) = new(z(n), z(n), z(n, 2), z(n), z(2), z(n, 2))
end
c(a) = fill!(a, 0f0)
clear!(a::Buffers) = (c(a.px); c(a.py); c(a.pxy); c(a.hc); c(a.hs); c(a.Is))

type ExtraTrees{T<:FloatingPoint}
	trees::Array{ExtremelyRandomizedTree{T}}
end

function ExtraTrees{T<:FloatingPoint}(data::Matrix{T}, labels; ntrees = 32, showprogress = true, pids = localworkers(), kargs...)
    if @unix ? (intersect(pids, localworkers()) == pids) : false
        workerpool = @p localworkers | unstack
        data = share(data)
        labels = share(labels)
        remotedata   = [data for i in 1:ntrees]
        remotelabels = [labels for i in 1:ntrees]
     else
        workerpool = @p workers | unstack
        remotedata   = @p map workerpool RemoteRef | map put! data
        remotelabels = @p map workerpool RemoteRef | map put! labels
    end

    ids = @p partsoflen (1:ntrees) len(workerpool)
    r = Any[]
    progress = showprogress && Progress(len(ids), 1, "Training ExtraTrees ... ", 50)
    for idset = ids
        refs = [@spawnat workerpool[i] ExtremelyRandomizedTree(fetch(remotedata[i]), fetch(remotelabels[i]); kargs...) for i in 1:length(idset)]
        push!(r, [fetch(ref) for ref in refs])
        showprogress && ProgressMeter.next!(progress)
    end
    @p flatten r | ExtraTrees{T}
end

function ExtraTrees{T<:Integer}(data::Matrix{T}, labels; kargs...)
	ExtraTrees(asfloat32(data), labels; kargs...)
end

function ExtremelyRandomizedTree{T1<:Number,T2<:Number}(data::AbstractArray{T1,2}, labels::AbstractArray{T2,2}; kargs...)

	assert(!isempty(data))
	assert(!isempty(labels))
	if size(data,2) != size(labels,2)
        error("ExtremelyRandomizedTree: size(data) was $(size(data)), size(labels) was $(size(labels))")
    end
	assert(!any(isnan(data)))

	ExtremelyRandomizedTree(buildSingleTree(data, labels; kargs...)...)
end

function swap(a, i::Int, j::Int)
    temp = a[i]
    a[i] = a[j]
    a[j] = temp
end

function halfsort!{T}(indices::Array{Int}, data::AbstractArray{T}, featureind::Int, threshold::T)
    assert(length(indices)>1)
    leftind = 1
    rightind = length(indices)
    while leftind <= rightind
        if data[featureind, indices[leftind]] < threshold
            leftind += 1
        else
            swap(indices, leftind, rightind)
            rightind -= 1
        end
    end
    view(indices, 1:leftind-1), view(indices, rightind+1:length(indices))
end

function splits!{T}(rsplits::AbstractArray{T}, mins::AbstractArray{T}, maxs::AbstractArray{T}, data::AbstractArray{T,2}, selectedfeatures::AbstractArray{Int}, indices::AbstractArray{Int})
    for m = selectedfeatures
        mins[m] = data[m, indices[1]]
        maxs[m] = data[m, indices[1]]
    end
    for n = indices, m = selectedfeatures
        mins[m] = min(mins[m], data[m, n])
        maxs[m] = max(maxs[m], data[m, n])
    end
    for m = selectedfeatures
        rsplits[m] = mins[m] + 100*eps(T) + rand(eltype(rsplits))*(maxs[m]-mins[m]-200*eps(T))
    end
end

function buildSingleTree(data, labels;
        classificationNMin::Int = 2,
        regressionNMin::Int = 5,
        regression::Bool = false,
        nmin::Int = regression ? regressionNMin : classificationNMin,
        nclasses::Int = asint(maximum(labels)),
        k::Int = round(Int,sqrt(size(data,1))),
	)

    if !regression && minimum(labels)<1
        error("labels need to be >= 1")
    end
    if !regression
        labels = asint(labels)
    end
        
    nLeafs = 0
    nNodes = 0
    initalSize = ceil(Int,length(labels)/100)
    leafsize = regression ? size(labels,1) : nclasses
    leafs = zeros(Float32, leafsize, initalSize)
    indmatrix = ones(Int, 4, initalSize)
    thresholds = ones(eltype(data), initalSize)
    indices = collect(1:len(data))
    mins = zeros(eltype(data), sizem(data))
    maxs = zeros(eltype(data), sizem(data))
    splits = zeros(eltype(data), sizem(data))
    buffers = Buffers(nclasses)

    function buildTree(indices)
        if len(indices) < nmin
			makeLeaf = true
        else
            makeLeaf = true
            for i in indices
                if labels[i] != labels[1]
                    makeLeaf = false
                    break
                end
            end
        end

        if !makeLeaf
			nonConstantFeatures = falses(sizem(data))
            for featureInd = 1:sizem(data)
                firstDatum = data[featureInd,indices[1]]
                for i = 2:len(indices)
                    if abs(data[featureInd,indices[i]] - firstDatum) > 200*eps(typeof(firstDatum))
                        nonConstantFeatures[featureInd] = true
                        break
                    end
                end
            end
            if !any(nonConstantFeatures)
				makeLeaf = true 
			end
        end
        
        nNodes = nNodes + 1
        if nNodes > len(indmatrix)
            indmatrix = hcat(indmatrix, indmatrix)
            thresholds = vcat(thresholds, thresholds)
        end
        nodeind = nNodes
        
        if makeLeaf
            if nLeafs == size(leafs,2)
                leafs = hcat(leafs, leafs)
            end
            nLeafs = nLeafs + 1

			if regression 
				leafs[:,nLeafs] = @p part labels indices | mean_
			else
				# return leaf label with class frequencies
				classhist = zeros(Float32, nclasses)
                s = zeroel(classhist)
				for i in indices
					classhist[labels[i]] += 1f0
                    s += 1f0
				end
                for i in 1:nclasses
                    leafs[i,nLeafs] = classhist[i]/s
                end
                # @show classhist s indices leafs
			end
           
            indmatrix[2, nodeind] = nodeind
            indmatrix[3, nodeind] = nodeind
            indmatrix[4, nodeind] = nLeafs
            thresholds[nodeind] = NaN
        else
            # 1) select random K from nonConstantFeatures

            nonConstantFeatures = find(nonConstantFeatures)
            randIndices = randperm(length(nonConstantFeatures)) 
            selectedFeatures = nonConstantFeatures[
                randIndices[1:min(k,length(nonConstantFeatures))]]
            assert(length(selectedFeatures)>0)
            # @show nonConstantFeatures selectedFeatures
            
            # 2) for each K:
            # 	randomSplit(data)
            # 		min and max von feature
            # 		return random split in [min,max]
            
            # mins = minimum(data[selectedFeatures,indices],2)
            # maxs = maximum(data[selectedFeatures,indices],2)
            # splits = rand(eltype(data), length(selectedFeatures),1).*(maxs-mins)+mins
            splits!(splits, mins, maxs, data, selectedFeatures, indices)
            
            # 3) over all K: computeScore, keep best split s*
            
            if length(indices) == 2
                bestSplitInd = @p randsample (1:length(selectedFeatures)) | fst
            else
                if length(selectedFeatures) == 1
                    bestSplitInd = 1
                else
                    bestSplitInd = rand(1:length(selectedFeatures))
                    scores = zeros(length(selectedFeatures))
                    for splitInd = 1:length(selectedFeatures)
                        scores[splitInd] = computeScore(regression, data[selectedFeatures[splitInd],indices], labels[:,indices], 
                            splits[selectedFeatures[splitInd]], nclasses, buffers)
                    end
                    temp = find(scores .== maximum(scores))
                    bestSplitInd = randsample(temp,1)[1]
                end
            end
            
            # 4) split data according to split s*, d1, d2
            featureind = selectedFeatures[bestSplitInd]
            threshold = splits[featureind]

            leftindices, rightindices = halfsort!(indices, data, featureind, threshold)
            # @show scores bestSplitInd splits leftindices rightindices

            indmatrix[1, nodeind] = featureind
            indmatrix[2, nodeind] = buildTree(leftindices)
            indmatrix[3, nodeind] = buildTree(rightindices)
            indmatrix[4, nodeind] = 0
            thresholds[nodeind] = threshold
        end
		nodeind
    end
	buildTree(indices)
	indmatrix = indmatrix[:, 1:nNodes]
	thresholds = thresholds[1:nNodes]
	leafs = leafs[:,1:nLeafs]
	(indmatrix, thresholds, leafs, regression)
end
 
function accumvotes!{T}(votesview::Matrix, leafind::Int, votesfor::Array{Int}, leafs::Array{T,2})
    # @show leafind
    for i in 1:length(votesfor)
        votesview[i] += leafs[votesfor[i], leafind]
    end
end

predict{T}(a::ExtraTrees{T}, data; kargs...) = predict(a, convert(Array{T,2}, data); kargs...)
function predict{T<:FloatingPoint}(a::ExtraTrees{T}, data::Array{T,2}; kargs...)
    predict!([], a, data; kargs... )
end

function predictitem{T}(dataview::Matrix{T}, indmatrix::Matrix{Int}, thresholds::Vector{T})
    nodeind = 0
    nextnodeind = 1

    while nextnodeind != nodeind
        nodeind = nextnodeind
        featureind = indmatrix[1,nodeind]::Int
        threshold = thresholds[nodeind]::T
        if dataview[featureind]::T < threshold
            nextnodeind = indmatrix[2,nodeind]::Int
        else
            nextnodeind = indmatrix[3,nodeind]::Int
        end
    end
    nodeind
end

function predicttree(extratree, data, dataview, votes, votesview, votesfor)
    indmatrix = extratree.indmatrix
    thresholds = extratree.thresholds
    for dataind = 1:len(data)
        view!(data, dataind, dataview)
        view!(votes, dataind, votesview)
        nodeind = predictitem(dataview, indmatrix, thresholds)
        # @show nodeind
        accumvotes!(votesview, indmatrix[4,nodeind], votesfor, extratree.leafs)
    end
end

function predict!{T<:Number}(votes::Array, a::ExtraTrees{T}, data::Array{T,2}; votesfor = [], returnvotes = false)
    assert(!isempty(data))
    assert(!isempty(a.trees))
    assert(eltype(data) == eltype(a.trees[1].thresholds))
               
    onlyvotes = true
    if isempty(votesfor)
        onlyvotes = false
        votesfor = collect(1:size(a.trees[1].leafs,1))
    end
    if isa(votesfor, Number)
        votesfor = [votesfor]
    end

    if isempty(votes)
        votes = zeros(eltype(a.trees[1].leafs), length(votesfor), len(data))
    else
        assert(size(votes)==(length(votesfor),len(data)))
    end

    dataview = view(data)
    votesview = view(votes)
    fill!(votes, zero(eltype(votes)))
 
    for extratree = a.trees
        predicttree(extratree, data, dataview, votes, votesview, votesfor)
    end

    factor = 1/length(a.trees)
    for i = 1:length(votes) votes[i] *= factor end

    if a.trees[1].regression
        return votes
    else
        if onlyvotes
            return votes
        else
            result = @p map votes indmax | row
            if returnvotes
                return result, votes
            else
                return result
            end
        end
    end
end
 
function volume!(a)
    if size(a,2)<2
        return 0.
    end
    mean_ = mean(a,2)
    for n = 1:size(a,2), m = 1:size(a,1)
        a[m,n] -= mean_[m]
    end
    c = cov(a')
    v = eigvals(c)
    prod(sqrt(max(0.001,v)))::Float64
end


function computeScore{T}(regression::Bool, featureData, labels::Matrix{T}, split, nclasses::Int, b)
	if regression
		ind = vec(featureData .< split)
		n = length(ind)
		nleft = sum(ind)
		nright = n-nleft

		if size(labels,1) == 1
 			v = var(labels)
			vleft = nleft > 1 ? var(labels[find(ind)]) : 0.
			vright = nright > 1 ? var(labels[find(!ind)]) : 0.
		else
			v = volume!(labels)
 			vleft = nleft > 1 ? volume!(labels[:,ind]) : 0.
			vright = nright > 1 ? volume!(labels[:,!ind]) : 0.
		end
		score = v > 0. ? (v - nleft/n*vleft - nright/n*vright)/v : 0.
	else
		x = labels
		y = [x < split ? 1 : 2 for x in featureData]

        clear!(b)
		for i = 1:length(x)
			b.pxy[x[i],y[i]] += 1
			b.px[x[i]] += 1
			b.py[y[i]] += 1
		end

        for i = 1:nclasses
            b.px[i] ./= length(x)
            b.py[i] ./= length(x)
            b.pxy[i,1] ./= length(x)
            b.pxy[i,2] ./= length(x)
        end

		for i = 1:nclasses
			b.hc[i] = b.px[i] == zeroel(b.px) ? zeroel(b.px) : b.px[i]*log(b.px[i])
		end
		for i = 1:2
			b.hs[i] = b.py[i] == zeroel(b.py) ? zeroel(b.py) : b.py[i]*log(b.py[i])
		end
		Hc = - sum(b.hc)
		Hs = - sum(b.hs)

		for xi = 1:nclasses
			for yi = 1:2
				b.Is[xi,yi] = b.pxy[xi,yi]*log( b.pxy[xi,yi]/(b.px[xi]*b.py[yi]))
			end
		end
        I = zeroel(b.Is)
        for i = eachindex(b.Is)
            !isnan(b.Is[i]) ? I += b.Is[i] : nothing
        end

		score = 2*I/(Hc + Hs)
	end
    assert(!isnan(score))
    score
end


end # module
