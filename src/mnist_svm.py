"""
mnist_svm
~~~~~~~~~

A classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier."""

#### Libraries
# My libraries
import mnist_loader 
# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

predictionSize = 200

def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()
    # train
    clf = svm.SVC()

    trainingSize = 2000
    clf.fit(training_data[0][:trainingSize], training_data[1][:trainingSize])
    # test
    predictions = [int(a) for a in clf.predict(test_data[0][:predictionSize])]
    num_correct = sum(int(a == y) for a, y in zip(predictions[:predictionSize], test_data[1][:predictionSize]))
    print "%s of %s values correct." % (num_correct, predictionSize)
    fig = plt.figure()
    pageIndex = 0
    drawFigure(fig, test_data, predictions, pageIndex)
    plt.show()
    
def drawFigure(fig, td, predictions, pageIndex):
    rows = 10
    columns = 20
    for plotIndex in range(0, rows * columns):
        totalIndex = rows * columns * pageIndex + plotIndex
        ax = fig.add_subplot(rows, columns, plotIndex+1, title = str(predictions[totalIndex]))
        ax.imshow(np.reshape(map(lambda x: [1-x,1-x,1-x] if predictions[totalIndex]==td[1][totalIndex] else [1,1-x,1-x], td[0][totalIndex]), (28, 28, 3)))
        ax.tick_params(axis='both', labelbottom='off', bottom='off', labeltop='off', top='off', labelleft='off', left='off', labelright='off', right='off')
    plt.tight_layout()
    plt.draw()
    if pageIndex * rows * columns < predictionSize:
        pageIndex += 1


if __name__ == "__main__":
    svm_baseline()
    