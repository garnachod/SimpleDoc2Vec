# Paragraph Vector Sentiment Classification IMDB

## Documentación:
* Articulo original
  * http://cs.stanford.edu/~quocle/paragraph_vector.pdf
* Articulo mejor explicado
  * http://arxiv.org/pdf/1411.2738v1.pdf
* Explicación con código sencillo
  * http://nbviewer.ipython.org/github/fbkarsdorp/doc2vec/blob/master/doc2vec.ipynb
  * http://linanqiu.github.io/2015/05/20/word2vec-sentiment/
  * https://districtdatalabs.silvrback.com/modern-methods-for-sentiment-analysis

## Método propuesto (Mejor aproximación a un problema real)
* Paso 0:
  * Limpiar conjunto de datos (StopWords, Caracteres raros...)
* Paso 1, generar vectores de las palabras:
  * Se utilizan los documentos de Train y Unsup (75K Docs) los otros 25k no se usan.
  * Se generar los entrenamientos que mejor resultado han dado por separado
* Paso 2:
  * Se generan los vectores de documentos a partir del entrenamiento anterior (Doc2Vec.infer_vector())
* Paso 3:
  * Se entrenan los clasificadores (50% train, 50% test, desglosados como en el conjunto inicial)

## Problemas encontrados y no solucionados
* Dada esta metodología, no es posible aproximarse a los resultados que se exponen en el paper
* Menor error encontrado 11.9% (DBOW, epocas 30, size 100)
* Los cosenos no se aproximan al entrenamiento o no tanto como se esperaría al menos en DM
* Utilizando todos los datos para entrenar y cogiendo los vectores generados en entrenamiento es posible bajar el error

## ¿Cómo hacerlo funcionar?
* Descargar conjunto de datos
  * http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
  * Guardar la carpeta descomprimida (aclImdb) en Data

* Ejecutar data/aclimdbCleaner.py (5 minutos de ejecución como mucho)
* Ejecutar doc2vec.py (En mi ordenador 40 minutos)
* Dos opciones de clasificación:
  * Ejecutar doc2vecClass.py (1 minuto como mucho)
  * Ejecutar doc2vecDBOWDMclass.py esta da la mejor precisión, junta DM y DBOW en un mismo vector de clasificación

## Librerias en uso
* Blas
* Numpy última versión
* Scipy (Cuidado) máxima versión 0.15
* Gensim última versión
* Sklearn

## Datos
Se debe simular el vector de los documentos (Método propuesto). En las siguentes tablas y gráficas se muestran los parámetros elegidos y el porque. Si el coseno se acerca a 1 es que ha "apendido" bien el vector documento.

![Alt text](./img/dmCosenos.png?raw=true "Tabla de cosenos")
![Alt text](./img/dmCosenoG.png?raw=true "DM gráfica cosenos")

Siguendo el mismo proceso con DBOW

![Alt text](./img/dbowCosenoG.png?raw=true "DBOW gráfica cosenos")

Si no se sigue el método propuesto y se aprenden todos los documentos, el menor error encontrado es del 1 - 0.88556

Continuaremos informando


