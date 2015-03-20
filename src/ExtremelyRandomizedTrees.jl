module ExtremelyRandomizedTrees

using FunctionalData

export ExtraTrees, predict

type ExtraTrees
	trees
end

immutable ExtremelyRandomizedTree{T}
	featureinds::Array{Int}
    thresholds::Array{T}
    leftinds::Array{Int}
    rightinds::Array{Int}
    leafinds::Array{Int}
	leafs::Array{T,2}
	maxdepth
	regression
end

function ExtraTrees(data, labels; ntrees = 32, kargs...)
	ExtraTrees([ExtremelyRandomizedTree(data, labels; kargs...) for i in 1:ntrees])
end

function ExtremelyRandomizedTree{T1<:Number,T2<:Number}(data::Array{T1,2}, labels::Array{T2,2}; kargs...)

	assert(!isempty(data))
	assert(!isempty(labels))
	assert(size(data,2) == size(labels,2))
	assert(!any(isnan(data)))
	assert(!any(isnan(data)))

	ExtremelyRandomizedTree(buildSingleTree(data, labels; kargs...)...)
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
    maxdepth = 0
    initalSize = iceil(length(labels)/101)
    leafsize = regression ? size(labels,1) : nclasses
    leafs = zeros(eltype(data), leafsize, initalSize)
    featureinds = ones(Int, initalSize)
    leftinds  = ones(Int, initalSize)
    rightinds = ones(Int, initalSize)
    leafinds = ones(Int, initalSize)
    thresholds = ones(eltype(data), initalSize)

    # format:  [featureInd threshold leftgotoInd rightgotoInd leafIndex]'


    function buildTree(data, labels, depth)
        # 	for all features
        # 		all data constant?
        # 		store indices to nonConstantFeatures
        #
        # length(data) < nMin ?
        #
        # all labels equal ?
        
        if size(data,2) < nmin
			makeLeaf = true
        else
            #     makeLeaf = true
            #     first = labels(1)
            #     for i = 2:length(labels)
            #         if labels(i)~=first
            #             makeLeaf = false
            #             break
            #         end
            #     end
            
            makeLeaf = !any(labels != labels[1])
        end
        if !makeLeaf
			nonConstantFeatures = falses(size(data,1))
            for featureInd = 1:size(data,1)
                firstDatum = data[featureInd,1]
                for i = 2:size(data,2)
                    if data[featureInd,i] != firstDatum
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
        if nNodes > length(featureinds)
            featureinds = vcat(featureinds, featureinds)
            thresholds = vcat(thresholds, thresholds)
            leftinds = vcat(leftinds, leftinds)
            rightinds = vcat(rightinds, rightinds)
            leafinds = vcat(leafinds, leafinds)
        end
        nodeInd = nNodes
        
        if makeLeaf
            if nLeafs == size(leafs,2)
                leafs = hcat(leafs, leafs)
            end
            nLeafs = nLeafs + 1

			if regression 
				leafs[:,nLeafs] = mean(labels,2)
			else
				# return leaf label with class frequencies
				classFrequencies = zeros(nclasses)
				for i = 1:nclasses
					classFrequencies[i] = sum(labels .== i)
				end
				leafs[:,nLeafs] = classFrequencies./sum(classFrequencies)
			end
           
            leftinds[nodeInd] = nodeInd
            rightinds[nodeInd] = nodeInd
            leafinds[nodeInd] = nLeafs
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
            
            mins = minimum(data[selectedFeatures,:],2)
            maxs = maximum(data[selectedFeatures,:],2)
            splits = rand(length(selectedFeatures),1).*(maxs-mins)+mins
            
            # 3) over all K: computeScore, keep best split s*
            
            if length(splits)>1
                scores = zeros(size(splits))
                for splitInd = 1:length(splits)
                    computeScore(splitInd) = 
                        computeScore(regression, data(selectedFeatures(splitInd),:), labels, splits(splitInd), nclasses)
                end
                bestSplitInd = indmax(scores)
            else
                bestSplitInd = 1
            end
            
            featureinds[nodeInd] = selectedFeatures[bestSplitInd]
            thresholds[nodeInd] = splits[bestSplitInd]  
            
            # 4) split data according to split s*, d1, d2
            dataind = 
                vec(data[selectedFeatures[bestSplitInd],:] .< splits[bestSplitInd])

            # 5) for both parts: t1=buildTree(d1), t2=buildTree(d2)
            maxdepth = max(depth,maxdepth)
            
            leftinds[nodeInd] = buildTree(data[:,dataind], labels[:,dataind], depth+1)
            rightinds[nodeInd] = buildTree(data[:,!dataind], labels[:,!dataind], depth+1)
            leafinds[nodeInd] = 0
        end
		nodeInd
    end
	buildTree(data, labels, 1)
	featureinds = featureinds[1:nNodes]
	thresholds = thresholds[1:nNodes]
	leftinds = leftinds[1:nNodes]
	rightinds = rightinds[1:nNodes]
	leafinds = leafinds[1:nNodes]
	leafs = leafs[:,1:nLeafs]
	(featureinds, thresholds, leftinds, rightinds, leafinds, leafs, maxdepth, regression)
end
 
function accumvotes!{T}(votesview::Array{T,2}, leafind::Int, votesfor::Array{Int}, leafs::Array{T,2})
    for i in votesfor
        votesview[i] += leafs[i, leafind]
    end
end

function getnextnodeind(dataview, nodeind::Int, extratree::ExtremelyRandomizedTree, thresholds)
    featureind = extratree.featureinds[nodeind]::Int
    threshold = thresholds[nodeind]::Float32
    if dataview[featureind]::Float32 < threshold
        extratree.leftinds[nodeind]::Int
    else
        extratree.rightinds[nodeind]::Int
    end
end

function predict{T<:Number}(a::ExtraTrees, data::Array{T,2}; votesfor::Array{Int} = collect(1:size(a.trees[1].leafs,1)), returnvotes = false)
    if isa(votesfor, Number)
        votesfor = [votesfor]
    end
    
    votes = zeros(eltype(a.trees[1].leafs), length(votesfor), len(data))
    predict!(votes, a, data, votesfor, returnvotes)
end

function predicttree(extratree, data, dataview, votes, votesview, votesfor)
    for dataind = 1:len(data)
        view!(data, dataind, dataview)
        view!(votes, dataind, votesview)

        nodeind = 0
        nextnodeind = 1

        while nextnodeind != nodeind
            nodeind = nextnodeind
            nextnodeind = getnextnodeind(dataview, nodeind, extratree, extratree.thresholds)::Int
        end
        accumvotes!(votesview, extratree.leafinds[nodeind], votesfor, extratree.leafs)
    end
end

function predict!{T<:Number}(votes, a::ExtraTrees, data::Array{T,2}, votesfor, returnvotes = false)
    assert(!isempty(data))
    assert(!isempty(a.trees))
    assert(eltype(data)==eltype(a.trees[1].thresholds))
    assert(eltype(data)==eltype(a.trees[1].leafs))

    dataview = view(data)
    votesview = view(votes)

    for extratree = a.trees
        predicttree(extratree, data, dataview, votes, votesview, votesfor)
    end

    votes ./= length(a.trees)
    if a.trees[1].regression
        votes
    else
        if votesfor != collect(1:size(a.trees[1].leafs,1))
            votes
        else
            result = zeros(1, size(data,2))
            for i = 1:size(data,2)
                result[i] = indmax(votes[:,i])
            end
            if returnvotes
                (result, votes)
            else
                result
            end
        end
    end
    votes
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
