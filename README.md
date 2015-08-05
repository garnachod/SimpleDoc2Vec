# SimpleDoc2Vec

## Documentación:
* Articulo original
  * http://cs.stanford.edu/~quocle/paragraph_vector.pdf
* Articulo mejor explicado
  * http://arxiv.org/pdf/1411.2738v1.pdf
* Explicación con código sencillo
  * http://nbviewer.ipython.org/github/fbkarsdorp/doc2vec/blob/master/doc2vec.ipynb
  * http://linanqiu.github.io/2015/05/20/word2vec-sentiment/
  * https://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis

## ¿Cómo hacerlo funcionar?
* Descargar conjunto de datos
  * http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
  * Guardar la carpeta descomprimida (aclImdb) en Data

* Ejecutar data/aclimdbCleaner.py (5 minutos de ejecución como mucho)
* Ejecutar doc2vec.py (En mi ordenador 40 minutos)
* Ejecutar doc2vecClass.py (1 minuto como mucho)

## Librerias en uso
* Blas
* Numpy última versión
* Scipy (Cuidado) máxima versión 0.15
* Gensim última versión
* NNet https://github.com/mczerny/NNet

## Pasos futuros:
* Intentar buscar un error de clasificación cercano al 7% que es el que se consigue en el paper
