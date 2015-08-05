# -*- coding: utf-8 -*-
from textblob import TextBlob
from nltk.stem.lancaster import LancasterStemmer
from os import walk
import codecs


"este fichero presupone que la carpeta aclImdb, está junto al fichero"

def cleanLine(line):
	line = line.replace("/", " ")
	line = line.replace("\\", " ")
	line = line.replace(".", " ")
	line = line.replace("-", " ")
	line = line.replace("'", " ")
	line = line.replace(":", " ")
	line = line.replace(u"’", " ")
	line = line.replace(u"‘", " ")
	line = line.replace("+", " ")
	line = line.replace("_", " ")
	line = line.replace(u"´", " ")
	line = line.replace(u"`", " ")
	line = line.replace(u"", " ")
	line = line.replace(u"", " ")
	line = line.replace(u"", " ")
	line = line.replace(u"", " ")
	line = line.replace(u"«", " ")
	line = line.replace(u"»", " ")
	line = line.replace(u"“", " ")
	line = line.replace(u"”", " ")
	line = line.replace(u"¨", " ")

	return line

if __name__ == '__main__':
	stopWords = "a|about|above|after|again|against|all|am|an|and|any|are|as|at|be|because|been|before|being|below|between|both|but|by|cannot|could|did|do|does|doing|down|during|each|few|for|from|further|had|has|have|having|he|her|here|hers|herself|him|himself|his|how|i|if|in|into|is|it|its|itself|me|more|most|my|myself|nor|not|of|off|on|once|only|or|other|ought|our|ours|ourselves|out|over|own|same|she|should|so|some|such|than|that|the|their|theirs|them|themselves|then|there|these|they|this|those|through|to|too|under|until|up|very|was|were|what|when|where|which|while|who|whom|why|with|would|you|your|yours|yourself|yourselves"
	st = LancasterStemmer()

	rootFolder = "aclImdb"
	generalFolders = ["test", "train"]
	classes = ["neg", "pos","unsup"]

	for folder in generalFolders:
		for clase in classes:
			fOut = codecs.open(folder+clase+".txt", "w", "utf-8")
			route = rootFolder+"/"+folder+"/"+clase
			try:
				for (dirpath, dirnames, filenames) in walk(route):
					for filename in filenames:
						fIn = codecs.open(route+"/"+filename, "r", "utf-8")
						palabrasOrdenadasFiltradas = []
						for line in fIn.readlines():
							line = cleanLine(line)
							textB = TextBlob(line)
							for palabra in textB.words:
								palabra = palabra.lower()
								if palabra in stopWords:
									continue
								#palabra = st.stem(palabra)
								if len(palabra) > 1:
									palabrasOrdenadasFiltradas.append(palabra)

							for palabra in palabrasOrdenadasFiltradas:
								fOut.write(palabra + " ")
						fIn.close()
						fOut.write("\n")
			except Exception, e:
				print e

			fOut.close()
