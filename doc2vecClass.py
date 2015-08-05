# classifier
from sklearn.linear_model import LogisticRegression
from gensim.models import Doc2Vec
import numpy
from sklearn import svm
from NNet import NeuralNet

if __name__ == '__main__':
	model = Doc2Vec.load('./imdb.d2v')

	#print model["TRAIN_POS_8029"]
	#exit()
	train_arrays = numpy.zeros((25000, 200))
	train_labels = numpy.zeros(25000)

	for i in range(12500):
	    prefix_train_pos = 'TRAIN_POS_' + str(i)
	    prefix_train_neg = 'TRAIN_NEG_' + str(i)
	    train_arrays[i] = model.docvecs[prefix_train_pos]
	    train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
	    train_labels[i] = 1
	    train_labels[12500 + i] = 0

	test_arrays = numpy.zeros((25000, 200))
	test_labels = numpy.zeros(25000)

	for i in range(12500):
	    prefix_test_pos = 'TEST_POS_' + str(i)
	    prefix_test_neg = 'TEST_NEG_' + str(i)
	    test_arrays[i] = model.docvecs[prefix_test_pos]
	    test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
	    test_labels[i] = 1
	    test_labels[12500 + i] = 0

	"""classifier = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
	gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
	shrinking=True, tol=0.001, verbose=False)

	classifier.fit(train_arrays, train_labels) 
	print "SVM"
	print classifier.score(test_arrays, test_labels)
	"""
	classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

	classifier.fit(train_arrays, train_labels)
	"""
	LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)"""
	print "Regresion logistica"
	print classifier.score(test_arrays, test_labels)




	nnet = NeuralNet(50, learn_rate=1e-2)
	maxiter = 1000
	nnet.fit(train_arrays, train_labels, fine_tune=False, SGD=True, batch=150, maxiter=maxiter, rho=0.9)
	print "Red neuronal"
	print nnet.score(test_arrays, test_labels)


