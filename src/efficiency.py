import mnist_loader

import matplotlib.pyplot as plt
import numpy as np
import math, random
import threading
from multiprocessing import Process, Queue
import time
from sklearn import svm

numRepetitions = 12 # Repeat experiment for the sake of precision
predictionSize = 150.
numProcessors = 4
plotInterval = 1

# Putting data sets in memory
training_data, validation_data, test_data = mnist_loader.load_data()
# Arrays containing results of proportion correct vs sample size given
q = Queue()
q.put([[],[]])
def iterate(times):
	for t in xrange(0,times):
		a = 10
		b = .01
		c = 0
		exp = lambda x: int(a ** ((x * b) + c))
		log = lambda x: int(math.log(x - c, a) / b)
		# sizes = map(exp, xrange(log(5), log(len(training_data[0]) - 1),20))
		sizes = map(exp, xrange(log(5), log(5000),plotInterval))
		for trainingSize in sizes:
			clf = svm.SVC()
			# Array to contain sampled images and answers
			sample = random.sample(xrange(0, len(training_data[0])), trainingSize)
			#sample = np.array(xrange(0, len(training_data[0][:trainingSize])))
			
			# Use algorithm to fit using only the sample
			clf.fit(training_data[0][sample], training_data[1][sample])
			# Make predictions
			sample = random.sample(xrange(0, len(test_data[0])), int(predictionSize))
			# sample = np.array(xrange(0, len(test_data[0][:predictionSize])))
			predictions = [int(i) for i in clf.predict(test_data[0][sample])]
			# Calculate proportion correct
			proportion = sum(int(predicted == actual) for predicted, actual in zip(predictions, test_data[1][sample])) / predictionSize
			print trainingSize, proportion
			results = q.get(True)
			if trainingSize in results[0]:
				results[1][results[0].index(trainingSize)].append(proportion)
				q.put(results)
			else:
				results[0].append(trainingSize)
				results[1].append([proportion])
				q.put(results)

proclist=[]
# For each single repetition:
for process in xrange(0, numProcessors):
	# Iterate through the options for our independent variable, the number of training instances available:
	# In this case, using a convenience sample and hoping that no arbitrary threshold of random training cases results in an unusual increase in prediction
	proclist.append(Process(target=iterate, args=[numRepetitions/numProcessors + ((numRepetitions % numProcessors) > process)]))
	proclist[process].start()
	proclist[process].join(0.1)

stop = False
while not stop:
	time.sleep(5)
	stop = True
	for p in proclist:
		if p.is_alive():
			stop = False

results = q.get(True)
plt.plot(results[0], map(lambda x: sum(x)/len(x), results[1]))
plt.show()