# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import codecs

if __name__ == '__main__':
	model = Doc2Vec.load('imdb_dm.d2v')
	print model.most_similar(positive=["man"], topn=10)
	"""fOut = codecs.open("diccionario.txt", "w", "utf-8")
	for key in model.vocab:
		fOut.write(key)
		fOut.write("\n")"""
	#print model.docvecs['TRAIN_POS_8029']