# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from collections import namedtuple
import time
import random
from blist import blist


# numpy
import numpy as np

# classifier
from sklearn.linear_model import LogisticRegression

class LabeledSentenceMio(namedtuple('LabeledSentenceMio', 'words tags')):
    def __new__(cls, words, tags):
        # add default values
        return super(LabeledSentenceMio, cls).__new__(cls, words, tags)

'''
class LabeledSentenceMio(namedtuple):
	"""docstring for LabeledSentenceMio"""
	def __init__(self, words=None, tags=None):
		super(LabeledSentenceMio, self).__init__()
		self.words = words
		self.tags = tags
		'''

class LabeledLineSentence(object):
	def __init__(self, sources):
		self.sources = sources
		self.sentences = None
		flipped = {}

		# make sure that keys are unique
		for key, value in sources.items():
			if value not in flipped:
				flipped[value] = [key]
			else:
				raise Exception('Non-unique prefix encountered')

	def to_array(self):
		if self.sentences is None:
			self.sentences = blist()
			for source, prefix in self.sources.items():
				with utils.smart_open(source) as fin:
					for item_no, line in enumerate(fin):
						#print [prefix + '_%s' % item_no]
						line = line.replace("\n", "")
						#exit()
						self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
						#self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
						#self.sentences.append(LabeledSentenceMio(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
						#print labe.tags
						#self.sentences.append(labe)
		#self.sentences)
		#perm = np.random.permutation(self.sentences.shape[0])
    	#model_dm.train(all_train_reviews[perm])
		return self.sentences
		

	def sentences_perm(self):
		random.shuffle(self.sentences)
		return self.sentences


if __name__ == '__main__':
	sources = {'data/trainneg.txt':'TRAIN_NEG', 'data/trainpos.txt':'TRAIN_POS', 'data/trainunsup.txt':'TRAIN_UNSP'}
	total_start = time.time()

	sentences = LabeledLineSentence(sources)
	model = Doc2Vec.load('./imdb_dbow.d2v')
	print "inicio vocab"
	sentences.to_array()
	print "fin vocab"
	#for sentence in sentences.sentences_perm():
	#	print sentence[1]

	for epoch in range(10):
		start = time.time()
		print "iniciando epoca DBOW:"
		model.train(sentences.sentences_perm())
		#model.train()
		end = time.time()
		print "tiempo de la epoca " + str(epoch) +": " + str(end - start)

	model.save('./imdb_dbow.d2v')

	"""
	model = Doc2Vec.load('./imdb_dm.d2v')
	#for sentence in sentences.sentences_perm():
	#	print sentence[1]

	for epoch in range(10):
		start = time.time()
		print "iniciando epoca DM:"
		model.train(sentences.sentences_perm())
		#model.train()
		end = time.time()
		print "tiempo de la epoca " + str(epoch) +": " + str(end - start)

	model.save('./imdb_dm.d2v')
	
	total_end = time.time()

	print "tiempo total:" + str((total_end - total_start)/60.0)
	"""