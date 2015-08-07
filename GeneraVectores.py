from gensim.models import Doc2Vec
import numpy as np
from numpy import dot
from gensim import utils, matutils


class GeneraVectores(object):
	"""docstring for GeneraVectores"""
	def __init__(self, model):
		super(GeneraVectores, self).__init__()
		self.model = model
		self.steps = 3
		self.alpha = 0.08
		self.docs = []
		self.lastFile = None

	
	def getVecsFromFile(self, file, max=-1):
		"""
			Retorna una lista de vectores inferidos
		"""
		if self.lastFile is None:
			self.lastFile = file

		if self.lastFile != file:
			self.docs = []

		self.lastFile = file

		vecsRetorno = None
		if len(self.docs) == 0:
			with utils.smart_open(file) as fin:
				for item_no, line in enumerate(fin):
					if max != -1 and item_no > max:
						break

					line = line.replace("\n", "")
					arrayWords = utils.to_unicode(line).split()
					self.docs.append(arrayWords)
					vecDoc = np.array([self.getVecsFromWords(arrayWords)])
					if vecsRetorno is None:
						vecsRetorno = vecDoc
					else:
						vecsRetorno = np.append(vecsRetorno, vecDoc, axis=0)
		else:
			for arrayWords in self.docs:
				vecDoc = np.array([self.getVecsFromWords(arrayWords)])
				if vecsRetorno is None:
					vecsRetorno = vecDoc
				else:
					vecsRetorno = np.append(vecsRetorno, vecDoc, axis=0)

		return vecsRetorno

	def getVecsFromWords(self, words):
		return self.model.infer_vector(words, steps=self.steps, alpha=self.alpha)


def pruebaCompletaCosenosDM():
	model = Doc2Vec.load('./imdb_dm.d2v')

	source = 'data/trainneg.txt'
	generador = GeneraVectores(model)


	steps = [1,2,3,5,7,10,15]
	alphas = [0.1, 0.075, 0.035]
	for alpha in alphas:
		for step in steps:
			generador.steps = step
			generador.alpha = alpha
			vecs = generador.getVecsFromFile(source)

			coseno_cum = 0.0
			for i in range(0, 12500):
				coseno_cum += dot(matutils.unitvec(vecs[i]), matutils.unitvec(model.docvecs["TRAIN_NEG_"+str(i)]))

			print "dm\t" + str(step) + "\t" + str(alpha) + "\t" + str((coseno_cum / 12500.0))

def pruebaCompletaCosenosDBOW():
	model = Doc2Vec.load('./imdb_dbow.d2v')

	source = 'data/trainneg.txt'
	generador = GeneraVectores(model)

	steps = [1,2,3,5,7,10,15]
	alphas = [0.1, 0.075, 0.035]
	for alpha in alphas:
		for step in steps:
			generador.steps = step
			generador.alpha = alpha
			vecs = generador.getVecsFromFile(source)

			coseno_cum = 0.0
			for i in range(0, 12500):
				coseno_cum += dot(matutils.unitvec(vecs[i]), matutils.unitvec(model.docvecs["TRAIN_NEG_"+str(i)]))

			print "dbow\t" + str(step) + "\t" + str(alpha) + "\t" + str((coseno_cum / 12500.0))


def puebaSimpleCosenos():
	model = Doc2Vec.load('./imdb_dm.d2v')

	source = 'data/trainneg.txt'
	generador = GeneraVectores(model)
	vecs = generador.getVecsFromFile(source)

	print "coseno primer vector, trainneg"
	print dot(matutils.unitvec(vecs[0]), matutils.unitvec(model.docvecs["TRAIN_NEG_0"]))

if __name__ == '__main__':
	pruebaCompletaCosenosDBOW()
