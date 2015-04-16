module ExtremelyRandomizedTrees

using FunctionalDataUtils, ProgressMeter

export ExtraTrees, predict, predict!

immutable ExtremelyRandomizedTree{T}
    indmatrix::Array{Int,2}
    thresholds::Array{T}
    leafs
    regression
end

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
        showprogress && next!(progress)
    end
    @p flatten r | ExtraTrees{T}
end

function ExtraTrees{T<:Integer}(data::Matrix{T}, labels; ntrees = 32, kargs...)
	ExtraTrees(float32(data), labels; kargs...)
end

function ExtremelyRandomizedTree{T1<:Number,T2<:Number}(data::AbstractArray{T1,2}, labels::AbstractArray{T2,2}; kargs...)

	assert(!isempty(data))
	assert(!isempty(labels))
	if size(data,2) != size(labels,2)
        error("ExtremelyRandomizedTree: size(data) was $(size(data)), size(labels) was $(size(labels))")
    end
	assert(!any(isnan(data)))
	assert(!any(isnan(data)))

	ExtremelyRandomizedTree(buildSingleTree(data, labels; kargs...)...)
end

function swap(a, i::Int, j::Int)
    temp = a[i]
    a[i] = a[j]
    a[j] = temp
end

function halfsort{T}(indices::Array{Int}, data::AbstractArray{T}, featureind::Int, threshold::T)
    assert(length(indices)>1)
    leftind = 1
    rightind = length(indices)
    while leftind < rightind-1
        lookingatind = indices[leftind]
        if data[featureind, lookingatind] < threshold
            leftind += 1
        else
            swap(indices, leftind, rightind)
            rightind -= 1
        end
    end
    view(indices, 1:leftind), view(indices, rightind:length(indices))
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
        rsplits[m] = mins[m] + rand(eltype(rsplits))*(maxs[m]-mins[m])
    end
end

function buildSingleTree(data, labels;
	classificationNMin = 2,
	regressionNMin = 5,
	regression = false,
	nmin = regression ? regressionNMin : classificationNMin,
	nclasses = int(maximum(labels)),
	k = round(sqrt(size(data,1))), 
	)

    if !regression && minimum(labels)<1
        error("labels need to be >= 1")
    end
        
    nLeafs = 0
    nNodes = 0
    initalSize = iceil(length(labels)/100)
    leafsize = regression ? size(labels,1) : nclasses
    leafs = zeros(Float32, leafsize, initalSize)
    indmatrix = ones(Int, 4, initalSize)
    thresholds = ones(eltype(data), initalSize)
    indices = collect(1:len(data))
    mins = zeros(eltype(data), sizem(data))
    maxs = zeros(eltype(data), sizem(data))
    splits = zeros(eltype(data), sizem(data))

    function buildTree(data, labels, indices)
        if len(indices) < nmin
			makeLeaf = true
        else
            makeLeaf = !any(labels != labels[indices[1]])
        end
        if !makeLeaf
			nonConstantFeatures = falses(sizem(data))
            for featureInd = 1:sizem(data)
                firstDatum = data[featureInd,indices[1]]
                for i = 2:len(indices)
                    if data[featureInd,indices[i]] != firstDatum
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
				classFrequencies = zeros(nclasses)
				for i = 1:nclasses
					classFrequencies[i] = sum(labels[indices] .== i)
				end
				leafs[:,nLeafs] = classFrequencies./sum(classFrequencies)
			end
           
            indmatrix[2, nodeind] = nodeind
            indmatrix[3, nodeind] = nodeind
            indmatrix[4, nodeind] = nLeafs
        else
            # 1) select random K from nonConstantFeatures

            nonConstantFeatures = find(nonConstantFeatures)
            randIndices = randperm(length(nonConstantFeatures)) 
            selectedFeatures = nonConstantFeatures[
                randIndices[1:min(k,length(nonConstantFeatures))]]
            
            # 2) for each K:
            # 	randomSplit(data)
            # 		min and max von feature
            # 		return random split in [min,max]
            
            # mins = minimum(data[selectedFeatures,indices],2)
            # maxs = maximum(data[selectedFeatures,indices],2)
            splits!(splits, mins, maxs, data, selectedFeatures, indices)
            # splits = rand(eltype(data), length(selectedFeatures),1).*(maxs-mins)+mins
            
            # 3) over all K: computeScore, keep best split s*
            
            if length(selectedFeatures)>1
                scores = zeros(selectedFeatures)
                for splitInd = 1:length(selectedFeatures)
                    computeScore(splitInd) = 
                        computeScore(regression, data(selectedFeatures(splitInd),indices), labels, 
                        splits[selectedFeatures[splitInd]], nclasses)
                end
                bestSplitInd = indmax(scores)
            else
                bestSplitInd = 1
            end
            
            indmatrix[1,nodeind] = selectedFeatures[bestSplitInd]
            thresholds[nodeind] = splits[bestSplitInd]  
            
            # 4) split data according to split s*, d1, d2
            featureind = selectedFeatures[bestSplitInd]
            threshold = splits[bestSplitInd]

            leftindices, rightindices = halfsort(indices, data, featureind, threshold)
            indmatrix[2, nodeind] = buildTree(data, labels, leftindices)
            indmatrix[3, nodeind] = buildTree(data, labels, rightindices)
            indmatrix[4, nodeind] = 0
        end
		nodeind
    end
	buildTree(data, labels, indices)
	indmatrix = indmatrix[:, 1:nNodes]
	thresholds = thresholds[1:nNodes]
	leafs = leafs[:,1:nLeafs]
	(indmatrix, thresholds, leafs, regression)
end
 
function accumvotes!{T}(votesview::Matrix, leafind::Int, votesfor::Array{Int}, leafs::Array{T,2})
    for i in 1:length(votesfor)
        votesview[i] += leafs[votesfor[i], leafind]
    end
end

predict{T<:Integer}(a::ExtraTrees, data::Array{T,2}; kargs...) = predict(a, float32(data); kargs...)
function predict{T<:FloatingPoint}(a::ExtraTrees{T}, data::Array{T,2}; votesfor = collect(1:size(a.trees[1].leafs,1)), returnvotes = false)
    if isa(votesfor, Number)
        votesfor = [votesfor]
    end
    
    votes = zeros(eltype(a.trees[1].leafs), length(votesfor), len(data))
    predict!(votes, a, data, votesfor, returnvotes)
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
        accumvotes!(votesview, extratree.indmatrix[4,nodeind], votesfor, extratree.leafs)
    end
end

function predict!{T<:Number}(votes, a::ExtraTrees{T}, data::Array{T,2}, votesfor = collect(1:size(a.trees[1].leafs,1)), returnvotes = false)
    assert(!isempty(data))
    assert(!isempty(a.trees))
    assert(eltype(data) == eltype(a.trees[1].thresholds))

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
        if votesfor != collect(1:size(a.trees[1].leafs,1))
            return votes
        else
            result = zeros(1, size(data,2))
            for i = 1:size(data,2)
                result[i] = indmax(votes[:,i])
            end
            if returnvotes
                return result, votes
            else
                return result
            end
        end
    end
end

 
function entropy(data, nclasses)
	px = zeros(nclasses)
	for i = 1:length(data)
		px[data[i]] = px[data[i]] + 1
	end
	px = px / sum(px)

	Hs = zeros(nclasses)
	for i = 1:length(px)
		Hs[i] = px[i]*log(px[i])
	end
	H = - sum(Hs)
end


function mutualInformation(x, y, nclasses)
	px = zeros(nclasses)
	py = zeros(nclasses)
	pxy = zeros(nclasses)
	for i = 1:length(x)
		pxy[x[i],y[i]] = pxy[x[i],y[i]] + 1
		px[x[i]] = px[x[i]] + 1
		py[y[i]] = py[y[i]] + 1
	end

	px = px/length(x)
	py = py/length(x)
	pxy = pxy/length(x)

	indI = 1
	Is = zeros(size(x))
	for xi = 1:nclasses
		for yi = 1:nclasses
			Is[indI] = pxy[xi,yi] * log(pxy[xi,yi] / (px[xi] * py[yi]))
			indI = indI + 1
		end
	end
	I = sum(Is[!isnan(Is)])
end
 
function volume(a)
    if size(a,2)<2
        return 0
    end
    a .-= mean(a,2)
    c = cov(a')
    v, _ = eig(c)
    prod(sqrt(max(0.001,v)))
end


function computeScore(regression, featureData, labels, split, nclasses)
	if regression
		ind = vec(featureData .< split)
		n = length(ind)
		nleft = sum(ind)
		nright = n-nleft
		assert(nleft>0)
		assert(nright>0)
		if size(labels,1) == 1
 			v = var(labels)
			vleft = var(labels[ind])
			vright = var(labels[!ind])
		else
			v = volume(labels)
 			vleft = volume(labels[:,ind])
			vright = volume(labels[:,!ind])
		end
		assert(v)>0
		score = (v - nleft/n*vleft - nright/n*vright)/v
	else
		#Hc = entropy(p, labels)
		#Hs = entropy(p, )
		#I  = mutualInformation(p, labels, 1+(featureData<split))

		x = labels
		y = 1+(featureData<split)

		px = zeros(nclasses)
		py = zeros(nclasses)
		pxy = zeros(nclasses)
		for i = 1:length(x)
			pxy[x[i],y[i]] = pxy[x[i],y[i]] + 1
			px[x[i]] = px[x[i]] + 1
			py[y[i]] = py[y[i]] + 1
		end
		#[px,py,pxy] = quickhist(nclasses, single(x), single(y))

		#assert(isequal(px,px2))
		#assert(isequal(py,py2))
		#assert(isequal(pxy,pxy2))

		px = px/length(x)
		py = py/length(x)
		pxy = pxy/length(x)

		hc = zeros(size(px))
		hs = zeros(1,2)
		for i = 1:length(px)
			hc[i] = px[i]*log(px[i])
		end
		for i = 1:2
			hs[i] = py[i]*log(py[i])
		end
		Hc = - sum(hc)
		Hs = - sum(hs)

		Is = zeros(nclasses,2)
		for xi = 1:nclasses
			for yi = 1:2
				Is[xi,yi] = pxy[xi,yi]*log( pxy[xi,yi]/(px[xi]*py[yi]))
			end
		end
		Is = vec(Is)
		I = sum(Is[!isnan(Is)])

		score = 2*I/(Hc + Hs)
	end
end


end # module
