# classifier
from sklearn.linear_model import LogisticRegression
from gensim.models import Doc2Vec
from GeneraVectores import GeneraVectores
import numpy as np
from sklearn import svm
from NNet import NeuralNet

if __name__ == '__main__':
	model_dbow = Doc2Vec.load('./imdb_dbow.d2v')
	model_dm = Doc2Vec.load('./imdb_dm.d2v')
	dim = 200
	#print model["TRAIN_POS_8029"]
	#exit()
	train_arrays = np.zeros((25000, dim))
	train_labels = np.zeros(25000)

	generador = GeneraVectores(model_dbow)
	dbowVecs_Pos = generador.getVecsFromFile("data/trainpos.txt")
	print "generados vectores dbowVecs_Pos"
	generador.setModel(model_dm)
	dmVecs_Pos = generador.getVecsFromFile("data/trainpos.txt")
	print "generados vectores dmVecs_Pos"
	generador.setModel(model_dbow)
	dbowVecs_Neg = generador.getVecsFromFile("data/trainneg.txt")
	print "generados vectores dbowVecs_Neg"
	generador.setModel(model_dm)
	dmVecs_Neg = generador.getVecsFromFile("data/trainneg.txt")
	print "generados vectores dmVecs_Neg"


	for i in range(12500):
		train_arrays[i] = np.concatenate((dbowVecs_Pos[i],dmVecs_Pos[i]))
		train_arrays[12500 + i] = np.concatenate((dbowVecs_Neg[i],dmVecs_Neg[i]))
		train_labels[i] = 1
		train_labels[12500 + i] = 0

	test_arrays = np.zeros((25000, dim))
	test_labels = np.zeros(25000)

	generador.setModel(model_dbow)
	dbowVecs_Pos = generador.getVecsFromFile("data/testpos.txt")
	print "generados vectores dbowVecs_Pos Test"
	generador.setModel(model_dm)
	dmVecs_Pos = generador.getVecsFromFile("data/testpos.txt")
	print "generados vectores dmVecs_Pos Test"
	generador.setModel(model_dbow)
	dbowVecs_Neg = generador.getVecsFromFile("data/testneg.txt")
	print "generados vectores dbowVecs_Neg Test"
	generador.setModel(model_dbow)
	dmVecs_Neg = generador.getVecsFromFile("data/testneg.txt")
	print "generados vectores dmVecs_Neg Test"

	for i in range(12500):
		test_arrays[i] = np.concatenate((dbowVecs_Pos[i],dmVecs_Pos[i]))
		test_arrays[12500 + i] = np.concatenate((dbowVecs_Neg[i],dmVecs_Neg[i]))
		test_labels[i] = 1
		test_labels[12500 + i] = 0
	"""
	classifier = svm.SVC()

	classifier.fit(train_arrays, train_labels)
	print "SVM"
	print classifier.score(test_arrays, test_labels)
	"""
	classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
		  intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

	classifier.fit(train_arrays, train_labels)
	print "Regresion logistica"
	print classifier.score(test_arrays, test_labels)



	
	nnet = NeuralNet(50, learn_rate=1e-2)
	maxiter = 5000
	nnet.fit(train_arrays, train_labels, fine_tune=False, SGD=True, batch=200, maxiter=maxiter)
	print "Red neuronal"
	print nnet.score(test_arrays, test_labels)
	
	"""
	print "imprimiendo .ARFF"
	f = open("entrenamiento.arff", "w", 4096)
	f.write("@RELATION dbow_dm")
	for i in range(dim):
		f.write("@ATTRIBUTE dbow_dm" + str(i)+ " NUMERIC\n")

	f.write("@ATTRIBUTE class {0,1}\n")

	f.write("@DATA\n")
	for indice, vector in enumerate(train_arrays):
		for elemento in vector:
			f.write(str(elemento) + ",")

		f.write(str(int(train_labels[indice])) + "\n")

	for indice, vector in enumerate(test_arrays):
		for elemento in vector:
			f.write(str(elemento) + ",")

		f.write(str(int(test_labels[indice])) + "\n")

	f.close()
	"""

