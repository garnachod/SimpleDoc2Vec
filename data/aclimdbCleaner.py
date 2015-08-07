# -*- coding: utf-8 -*-
from textblob import TextBlob
from nltk.stem.lancaster import LancasterStemmer
from os import walk
import codecs
import re


"este fichero presupone que la carpeta aclImdb, está junto al fichero"

prog = re.compile("[0-9]+")

def cleanLine(line):
	line = line.replace("?", " QUESTION ")
	line = line.replace("!", " EXCLAMATION ")
	line = line.replace("...", " DOTDOTDOT ")
	#line = line.replace(".", " DOT ")
	#line = line.replace(",", " COMMA ")
	#line = prog.sub(" ", line)
	line = line.replace(".", " ")
	line = line.replace(",", " ")
	line = line.replace("/", " ")
	line = line.replace("\\", " ")
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
	stopWords = {"a": 1,"about": 1,"above": 1,"after": 1,"again": 1,"against": 1,"all": 1,"am": 1,"an": 1,"and": 1,"any": 1,"are": 1,"as": 1,"at": 1,"be": 1,"because": 1,"been": 1,"before": 1,"being": 1,"below": 1,"between": 1,"both": 1,"but": 1,"by": 1,"could": 1,"did": 1,"do": 1,"does": 1,"doing": 1,"down": 1,"during": 1,"each": 1,"few": 1,"for": 1,"from": 1,"further": 1,"had": 1,"has": 1,"have": 1,"having": 1,"he": 1,"her": 1,"here": 1,"hers": 1,"herself": 1,"him": 1,"himself": 1,"his": 1,"how": 1,"i": 1,"if": 1,"in": 1,"into": 1,"is": 1,"it": 1,"its": 1,"itself": 1,"me": 1,"more": 1,"most": 1,"my": 1,"myself": 1,"nor": 1,"of": 1,"off": 1,"on": 1,"once": 1,"only": 1,"or": 1,"other": 1,"ought": 1,"our": 1,"ours": 1,"ourselves": 1,"out": 1,"over": 1,"own": 1,"same": 1,"she": 1,"should": 1,"so": 1,"some": 1,"such": 1,"than": 1,"that": 1,"the": 1,"their": 1,"theirs": 1,"them": 1,"themselves": 1,"then": 1,"there": 1,"these": 1,"they": 1,"this": 1,"those": 1,"through": 1,"to": 1,"too": 1,"under": 1,"until": 1,"up": 1,"very": 1,"was": 1,"were": 1,"what": 1,"when": 1,"where": 1,"which": 1,"while": 1,"who": 1,"whom": 1,"why": 1,"with": 1,"would": 1,"you": 1,"your": 1,"yours": 1,"yourself": 1,"yourselves":1}
	
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
							line = line.lower()
							line = cleanLine(line)
							textB = TextBlob(line)
							for palabra in textB.words:
								if palabra in stopWords:
									continue
								if palabra == "br":
									continue
								#palabra = st.stem(palabra)
								if len(palabra) > 1:
									palabrasOrdenadasFiltradas.append(palabra)

						if len(palabrasOrdenadasFiltradas) < 9:
							for i in range(len(palabrasOrdenadasFiltradas), 10):
								fOut.write("NULL" + " ")

						for palabra in palabrasOrdenadasFiltradas:
							fOut.write(palabra + " ")

						

						fIn.close()
						fOut.write("\n")
			except Exception, e:
				print e

			fOut.close()
