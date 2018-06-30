
# coding: utf-8

# # Hot500 song recommendations
# 
# Build song recommendation out of the k=100 nearest neighbors

# In[1]:


from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import pandas as pd
import numpy as np
import matplotlib as plt
import importlib
from pyspark.ml.feature import Tokenizer, CountVectorizer, MinHashLSH
from pyspark.sql.types import IntegerType, StringType, ArrayType

import mpd


# In[2]:


# Will allow us to embed images in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
# change default plot size
plt.rcParams['figure.figsize'] = (15,10)


# ## Load and prep data
# 
# * Load the full data set
# * Load the picked k=100 approx Nearest Neighbor results
# * Build song recommdations based on songs in nearest playlist

# In[3]:


mpd_all=mpd.load(spark, "onebig", 1)


# Get the ranked popularity of songs in the mpd.

# In[32]:


cv = CountVectorizer(inputCol="track_uri", outputCol="features", minDF=2, vocabSize=2000000)


# In[33]:


model=cv.fit(mpd_all.select("pid", "tracks.track_uri"))


# In[35]:


result=model.transform(mpd_all.select("pid", "tracks.track_uri"))


# In[36]:


#model, result = mpd.vectorizecol(mpd_all.select("pid", "tracks.track_uri"), "track_uri", "features", 2000000)


# In[37]:


result.printSchema()


# In[38]:


result.count()


# In[121]:


importlib.reload(mpd)


# In[122]:


vdf = mpd.buildvocabdf(spark, model.vocabulary)


# In[8]:


vdf.show(5)


# In[126]:


vdf.describe("tid").show()


# In[42]:


vdf.printSchema()


# In[40]:


vdf.count()


# Get the Hot100 playlists that match the challenge set.

# In[10]:


hot100 = spark.createDataFrame(pd.read_pickle("ex-neighborpl.pkl"))


# In[132]:


hot100 = spark.createDataFrame(pd.read_pickle("neighborpl.pkl"))


# In[133]:


hot100.orderBy("pid").show(5)


# In[52]:


arraylength = f.udf(lambda x: len(x), IntegerType())


# In[134]:


h100cnt = hot100.withColumn("reclen", arraylength(hot100.recpl))


# In[135]:


h100cnt.orderBy("reclen").show()


# In[157]:


h100cnt.groupBy("reclen").count().orderBy("reclen").show(5)


# In[136]:


h100cnt.orderBy("reclen").groupBy("reclen").count().describe("count").show()


# In[137]:


h100cnt.describe("reclen").show()


# We can see that most results will have gotten 100 neighbors

# In[138]:


mpd.plothist(h100cnt, "reclen", 11)


# In[139]:


h100 = hot100.select("pid", f.explode("recpl").alias("recpid"))


# In[140]:


h100withtracks = h100.join(result, result.pid == h100.recpid).drop(result.pid).drop(result.features).orderBy("pid")


# In[43]:


h100withtracks.show(5)


# ## Get the ranked resutls of tracks from the recommended neighboring playlists.

# In[141]:


trackrank = h100withtracks.select("pid", f.explode("track_uri").alias("track")).groupBy("pid","track").count().sort(f.desc("count"))


# In[158]:


trackrank.orderBy("pid", f.desc("count")).show(5)


# In[149]:


trackrank.printSchema()


# ### Exlore a single playlist

# In[163]:


testpid = 1000061


# In[164]:


trackrank.where(f.col("pid") == testpid).show()


# In[160]:


trackrank.where(f.col("pid") == testpid).count()


# Add the global rank

# In[161]:


grank=trackrank.join(vdf, trackrank.track == vdf.term).drop(vdf.term)


# In[81]:


grank.printSchema()


# Here is the track recommendation for one playlist based on the popularity of the track in the neighborhood with additional sorting by the globab popularity. Global popularity is based on count vecorizer with most popular recieving the lowest value.

# In[162]:


grank.where(f.col("pid") == testpid).orderBy(f.desc("count"), f.asc("tid")).show()


# ## Eliminate tracks included in the search

# In[83]:


mpd_test=spark.read.json("../mpd-challenge/challenge_set.json", multiLine=True)


# In[84]:


cpl=mpd_test.select(f.explode("playlists").alias("playlist"))


# In[85]:


recdf=cpl.select("playlist.pid", "playlist.tracks")


# In[97]:


recdf.describe("pid").show()


# test a playlist

# In[181]:


existingtracks = recdf.where(recdf.pid == testpid).select(f.explode("tracks.track_uri").alias("track"))


# In[182]:


existingtracks.printSchema()


# In[183]:


existingtracks.show()


# In[195]:


existingtracks.toPandas()["track"].tolist()


# In[169]:


grank.where(f.col("pid") == testpid).where(~grank.track.isin(existingtracks.toPandas()["track"].tolist())).show()


# ## Iterate over search results and provide track list

# In[184]:


def gettracks(chpl, grank, recdf):
    # get the challenge playlist id
    testpid = chpl.pid
    
    # get the provided tracks
    existingtracks = recdf.where(recdf.pid == testpid).select(f.explode("tracks.track_uri").alias("track"))
    
    # get the tracks from the global rank
    df = grank.where(f.col("pid") == testpid).where(~grank.track.isin(existingtracks.toPandas()["track"].tolist()))
    
    tracklist = df.orderBy(f.desc("count"), f.asc("tid")).toPandas()["track"].tolist()
    
    recommend = { "pid": testpid, "tracks": [tracklist]}
    #print("DEBUG: " + testpid + " " + pidlist)
    
    return recommend


# In[185]:


recommended = pd.DataFrame({"pid":0, "tracks":[]})


# In[186]:


for row in hot100.limit(10).rdd.collect():
    rec = gettracks(row, grank, recdf)
    recommended = recommended.append(pd.DataFrame(rec))


# In[187]:


recommended


# In[194]:


recommended["tracks"].apply(lambda x: len(x))


# In[196]:


recommended.to_pickle("rectracks.pkl")


# In[197]:


recommended.to_csv("rectracks.csv")

