# generic pyspark and jupyter notebook initialization
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import pandas as pd
import matplotlib as plt
from pyspark.sql.window import Window as W

# import for utilies in this module
import os
from pyspark.sql.functions import explode
from pyspark.sql.types import StructField, StructType, LongType, StringType
from pyspark.ml.feature import CountVectorizer
import pyspark.ml.feature as mlf

def load(spark, dir, limit):
   """ load the mpd dataset from dir into spark, limit files by count """

   filenames = os.listdir(dir)
   count = 0
   for filename in sorted(filenames):
      if filename.endswith(".json"):
         newDF =  spark.read.json(dir + "/" + filename)
         if count > 0:
            df = df.union(newDF)
         else:
            df = newDF
         count += 1
         if count > limit:
           break

   return df

def playlist_flatten(df):
   """ create a flatten playlist with all tracks as fields """

   pDF=df.select("pid", explode("tracks").alias("track")).select("pid", "track.*")
   return pDF

def create_dict(df):
   """ create a dictionary with a unique id for the input of distinct records"""

   newSchema = StructType([StructField("id", LongType(), False)]
                       + df.schema.fields)

   df=df.rdd.zipWithIndex()\
                      .map(lambda row, id: {k:v
                                              for k, v
                                              in row.asDict().items() + [("id", id)]})\
                      .toDF(newSchema)

   return df 

def create_taa_vec(pdf, tdict, aldict, ardict):
   """ create playlist, track, album, artist dataframe (vector)

   use the repesctive dictionaries to translate the uri strings to 
   integers creating a 3D space for playlists
   """
   pdf.createOrReplaceTempView("playlist")
   tdict.createOrReplaceTempView("track")
   aldict.createOrReplaceTempView("album")
   ardict.createOrReplaceTempView("artist")

   pvec=spark.sql("Select \
                     playlist.pid, track.id as tid, album.id as alid, artist.id as arid \
                   from \
                     playlist, track, album, artist \
                   where \
                         playlist.track_uri = track.track_uri \
                     and playlist.album_uri = album.album_uri \
                     and playlist.artist_uri = artist.artist_uri")

   return pvec

def plothist(df, col, buckets):
   """
   Doing the heavy lifting in Spark. We could leverage the `histogram` function from the RDD api

   this comes from:
   https://stackoverflow.com/a/39891776
   """
   histogram = df.select(col).rdd.flatMap(lambda x: x).histogram(buckets)

   # Loading the Computed Histogram into a Pandas Dataframe for plotting
   pd.DataFrame(
      list(zip(*histogram)),
      columns=['bin', 'frequency']
   ).set_index(
      'bin'
   ).plot(kind='bar');

def vectorizecol(df, incol, outcol, size=1<<18):
   """
   Vectorize a column of terms and add it to the dataframe
   return df and model
   """
   cv = CountVectorizer(inputCol=incol, outputCol=outcol, vocabSize=size)
   model = cv.fit(df)
   result = model.transform(df)

   return model, result

def buildvocabdf(spark, vocabulary):
   """
   build a vocabulary dataframe with id using list of input words
   useful for joining with results
   """

   # note: trying to provide the column name in the schema but get errors
   # opt for renamed workaround
   #schema = StructType([StructField("term", StringType(), nullable=False)])
   #df = spark.createDataFrame(vocabulary, schema)
   df = spark.createDataFrame(vocabulary, StringType())
   df = df.withColumnRenamed("value", "term")
   df = df.withColumn("mid", f.monotonically_increasing_id())

   # build increase by one id
   windowSpec = W.orderBy("mid")
   df = df.withColumn("tid", f.row_number().over(windowSpec)).drop("mid")

   return df

def scatterplotfreq(dfcountcol):
   """
   input pandas dataframe of counts to plot the frequency of
   """

   Y=dfcountcol.select("count").toPandas()
   X=pd.DataFrame({'X': range(1,Y.size+1,1)})

   plt.pyplot.scatter(X,Y)

def canonicaltokens(df, inputColumn, outputColumn):
   """
   turn input column of strings into canonical format as output column of tokens
   return as output column added to the dataframe
   """

   newname = df.withColumn("cleanname", \
       f.regexp_replace(f.regexp_replace(f.rtrim(f.ltrim(f.col(inputColumn))), \
       " (\w) (\w) ", "$1$2"), "(\w) (\w) (\w)$", "$1$2$3"))

   newtokenizer = mlf.Tokenizer(inputCol="cleanname", outputCol="words")
   chtokenized = newtokenizer.transform(newname).drop("cleanname")

   stopwordremover = mlf.StopWordsRemover(inputCol="words", outputCol=outputColumn)
   canonicalname = stopwordremover.transform(chtokenized).drop("words")

   return canonicalname
