import csv
import random
import math

def loadCSV(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summarise = {}
    for classValue, instance in separated.iteritems():
        summarise[classValue] = summarize(instance)
    return summarise


def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x,mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProbe = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProbe:
            bestProbe = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct+=1
    return (correct/float(len(testSet)))*100.0



def test():
    dataset = [[1],[2],[3],[4],[5]]
    splitRatio = 0.67
    train, test = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train with {1} and test with {2}'.format(len(dataset), train, test))
    dataset = [[1,20,1], [2,21,0], [3,22,1]]
    separated = separateByClass(dataset)
    print('Separated instances: {0}').format(separated)
    numbers = [1,2,3,4,5]
    print('Summary of {0}: mean={1}, stdev={2}').format(numbers, mean(numbers), stdev(numbers))
    dataset = [[1,20,0], [2,21,1], [3,22,0]]
    summary = summarize(dataset)
    print('Attribute summaries: {0}').format(summary)
    dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
    summary = summarizeByClass(dataset)
    print('Summary by class value: {0}').format(summary)
    summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
    inputVector = [1.1, '?']
    probabilities = calculateClassProbabilities(summaries, inputVector)
    print('Probabilities for each class: {0}').format(probabilities)
    summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
    inputVector = [1.1, '?']
    result = predict(summaries, inputVector)
    print('Prediction: {0}').format(result)
    summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
    testSet = [[1.1, '?'], [19.1, '?']]
    predictions = getPredictions(summaries, testSet)
    print('Predictions: {0}').format(predictions)
    testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
    predictions = ['a', 'a', 'a']
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}').format(accuracy)


def main():
    filename = 'data.csv'
    splitRatio = 0.67
    dataset = loadCSV(filename)
    print('loaded data file {0} with {1} rows').format(filename, len(dataset))
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
    summaries = summarizeByClass(trainingSet)
    #test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%').format(accuracy)


main()

# test()

