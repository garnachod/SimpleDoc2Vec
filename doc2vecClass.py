# classifier
from sklearn.linear_model import LogisticRegression
from gensim.models import Doc2Vec
import numpy
from GeneraVectores import GeneraVectores
from sklearn import svm
from NNet import NeuralNet

if __name__ == '__main__':
	model = Doc2Vec.load('./imdb_dbow.d2v')

	#print model["TRAIN_POS_8029"]
	#exit()
	dim = 100
	train_arrays = numpy.zeros((25000, dim))
	train_labels = numpy.zeros(25000)

	generador = GeneraVectores(model)
	Pos = generador.getVecsFromFile("data/trainpos.txt")
	print "generados vectores Pos"
	Neg = generador.getVecsFromFile("data/trainneg.txt")
	print "generados vectores Neg"

	for i in range(12500):
	    train_arrays[i] = Pos[i]
	    train_arrays[12500 + i] = Neg[i]
	    train_labels[i] = 1
	    train_labels[12500 + i] = 0

	test_arrays = numpy.zeros((25000, dim))
	test_labels = numpy.zeros(25000)

	Pos = generador.getVecsFromFile("data/testpos.txt")
	print "generados vectores Pos TEST"
	Neg = generador.getVecsFromFile("data/testneg.txt")
	print "generados vectores Neg TEST"

	for i in range(12500):
	    test_arrays[i] = Pos[i]
	    test_arrays[12500 + i] = Neg[i]
	    test_labels[i] = 1
	    test_labels[12500 + i] = 0


	classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

	classifier.fit(train_arrays, train_labels)

	print "Regresion logistica"
	print classifier.score(test_arrays, test_labels)



