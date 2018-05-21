import os
from pyspark.sql.functions import explode
from pyspark.sql.types import StructField, StructType, LongType

def load(spark, dir, limit):
   """ load the mpd dataset from dir into spark, limit files by count """

   filenames = os.listdir(dir)
   count = 0
   for filename in sorted(filenames):
      if filename.startswith("spark.mpd.slice.") and filename.endswith(".json"):
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
