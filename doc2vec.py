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
						line = line.replace("\n", "")
						self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
						
		return self.sentences
		

	def sentences_perm(self):
		random.shuffle(self.sentences)
		return self.sentences


if __name__ == '__main__':
	sources = {'data/trainneg.txt':'TRAIN_NEG', 'data/trainpos.txt':'TRAIN_POS', 'data/trainunsup.txt':'TRAIN_UNSP'}
	dimension = 100
	total_start = time.time()

	sentences = LabeledLineSentence(sources)
	dbow = True
	if dbow:
		model = Doc2Vec(min_count=1, window=10, size=dimension, sample=1e-3, negative=5, dm=0 ,workers=6, alpha=0.04)
		
		print "inicio vocab"
		model.build_vocab(sentences.to_array())
		print "fin vocab"
		first_alpha = model.alpha
		last_alpha = 0.01
		next_alpha = first_alpha
		epochs = 30
		for epoch in range(epochs):
			start = time.time()
			print "iniciando epoca DBOW:"
			print model.alpha
			model.train(sentences.sentences_perm())
			end = time.time()
			next_alpha = (((first_alpha - last_alpha) / float(epochs)) * float(epochs - (epoch+1)) + last_alpha)
			model.alpha = next_alpha
			print "tiempo de la epoca " + str(epoch) +": " + str(end - start)

		model.save('./imdb_dbow.d2v')

	dm = True
	if dm:
		#model = Doc2Vec(min_count=1, window=10, size=dimension, sample=1e-3, negative=5, workers=6, dm_mean=1, alpha=0.04)
		model = Doc2Vec(min_count=1, window=10, size=dimension, sample=1e-3, negative=5, workers=6, alpha=0.04)
		#model = Doc2Vec(min_count=1, window=10, size=dimension, sample=1e-3, negative=5, workers=6, alpha=0.04, dm_concat=1)
		#
		print "inicio vocab"
		model.build_vocab(sentences.to_array())
		print "fin vocab"
		first_alpha = model.alpha
		last_alpha = 0.01
		next_alpha = first_alpha
		epochs = 30
		for epoch in range(epochs):
			start = time.time()
			print "iniciando epoca DM:"
			print model.alpha
			model.train(sentences.sentences_perm())
			end = time.time()
			next_alpha = (((first_alpha - last_alpha) / float(epochs)) * float(epochs - (epoch+1)) + last_alpha)
			model.alpha = next_alpha
			print "tiempo de la epoca " + str(epoch) +": " + str(end - start)

		model.save('./imdb_dm.d2v')
	
	total_end = time.time()

	print "tiempo total:" + str((total_end - total_start)/60.0)