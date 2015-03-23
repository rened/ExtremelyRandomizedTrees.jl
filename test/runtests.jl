println("Starting runtests.jl ...")

using ExtremelyRandomizedTrees
using FactCheck

doPlot = false

if doPlot
	using PyPlot
end

doDummyClassification = true
doDummyRegression = true
doClassification = true
doRegression1to1 = true
doRegression1to2 = true
doRegression2to2 = true
testIndivProbPrediction = true

if doDummyClassification
    data = [-2 -1 1 2]
    labels = [1 1 2 2]
    a = ExtraTrees(float(data), labels)

    testdata = linspace(-5,5,100)'

    result, votes = predict(a, testdata, returnvotes = true)
    doPlot && plot(testdata', [votes' result'])
end

if doDummyRegression
    trainingData = [-10:10]'
    trainingLabels = trainingData+100
    testData = trainingData

    a = ExtraTrees(trainingData, trainingLabels; regression = true)
    result = predict(a,testData)

    doPlot && plot(testData', result', "b-")
end

if doClassification
    trainingData = hcat(rand(2,100), rand(2,100)+1, rand(2,100)+2)
    trainingLabels = hcat(ones(1,100), 2*ones(1,100), 3*ones(1,100))

    a = ExtraTrees(trainingData, trainingLabels)

    X = [x for x in -5:0.1:10, y in -5:0.1:10]
    Y = [y for x in -5:0.1:10, y in -5:0.1:10]

    result = predict(a, [vec(X) vec(Y)]');
    
    doPlot && imshow(reshape(result, size(X)))
end

####################################
# univariate regression
# 1 feature, 1 target
####################################

if doRegression1to1
    trainingData = linspace(0,2*pi,100)'
    testData = linspace(0,2*pi,1000)'
    trainingLabels = 10 + sin(trainingData) + (0.3*rand(size(trainingData))-0.15)

    a = ExtraTrees(trainingData, trainingLabels; regression = true)
    result = predict(a,testData)

	if doPlot
		plot(trainingData, trainingLabels,"b.", hold = true)
		plot(testData', result',"g-", hold = true)
		title("univariate regression - 1 feature, 1 target")
	end
end

if testIndivProbPrediction
    trainingData = hcat(rand(2,100), rand(2,100)+1, rand(2,100)+2)
    trainingLabels = hcat(ones(1,100), 2*ones(1,100), 3*ones(1,100))

    a = ExtraTrees(trainingData, trainingLabels)

    X = [x for x in -5:0.1:10, y in -5:0.1:10]
    Y = [y for x in -5:0.1:10, y in -5:0.1:10]

    result, votes = predict(a, [vec(X) vec(Y)]', returnvotes = true)
    indivvotes = Array(Any,3)
    for i = 1:3
        indivvotes[i] = predict(a, [vec(X) vec(Y)]'; votesfor=i)
        #assert(all(votes[i,:]/sum(votes[i,:]) == indivvotes[i]))
    end    
	if doPlot
		for i = 1:3
			subplot(1,3,i); imshow(reshape(indivvotes[i], size(X)))
		end
	end
end

####################################
# multivariate regression
# 1 feature, 2 target values
####################################

if doRegression1to2
    trainingData = linspace(0,2*pi,100)'
    testData = linspace(0,2*pi,1000)'
    trainingLabels = [3 + sin(trainingData) + (0.3*rand(size(trainingData))-0.15);
        10 + trainingData.^2 + (0.3*rand(size(trainingData))-0.15)]

    a = ExtraTrees(trainingData, trainingLabels; regression = true)
    result = predict(a,testData)

	if doPlot
		plot(trainingData, trainingLabels[1,:],"b.", hold = true)
		plot(trainingData, trainingLabels[2,:],"b.", hold = true)
		plot(testData', result[1,:]',"g-", hold = true)
		plot(testData', result[2,:]',"g-", hold = true)
		title("multivariate regression - 1 feature, 2 target values")
	end
end

####################################
# multivariate regression
# 2 features, 2 target values
####################################

if doRegression2to2
    trainingData = linspace(0,2*pi,100)'
    testData = linspace(0,2*pi,1000)'
    trainingTargets = vcat([10 + sin(trainingData) + (0.3*rand(size(trainingData))-0.15)],
    [10 + cos(trainingData) + (0.3*rand(size(trainingData))-0.15)]);
    testTargets = vcat([10 + sin(testData)],
    [10 + cos(testData)]);

    a = ExtraTrees(trainingData, trainingTargets; regression = true)
    result = predict(a,testData)

    assert(maximum(abs(result-testTargets))<0.3)
end


println("  ... runtests.jl done!")
