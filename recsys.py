
# coding: utf-8

# # Build song recommendations

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
# * Vectorize the playlists into sparse vectors
# * Extract the vocabulary with tid to allow translation back to track_uri

# In[3]:


mpd_all=mpd.load(spark, "onebig", 1)


# ## Build track, artist and name features

# The track_uri and artist_uri columns are already lists and can be processed in their raw format.

# In[4]:


trackdf = mpd_all.select("pid", "tracks.track_uri")


# In[5]:


artistdf = mpd_all.select("pid", "tracks.artist_uri")


# In[6]:


namedf = mpd.canonicaltokens(mpd_all.select("pid", "name").fillna({"name": ""}), "name", "filtered").drop("name")


# In[7]:


mergeCols = f.udf((lambda x, y: x + y), ArrayType(StringType()))


# In[8]:


featuredf = trackdf.join(artistdf, trackdf.pid == artistdf.pid).drop(artistdf.pid)


# In[9]:


featuredf = featuredf.withColumn("tafeatures", mergeCols("track_uri", "artist_uri")).drop("track_uri").drop("artist_uri")


# In[10]:


featuredf.printSchema()


# In[11]:


featuredf = featuredf.join(namedf, namedf.pid == featuredf.pid).drop(namedf.pid)


# In[12]:


featuredf = featuredf.withColumn("features", mergeCols("filtered", "tafeatures")).drop("tafeatures").drop("filtered")


# In[13]:


featuredf.printSchema()


# In[14]:


featuredf.show(5)


# ## Build feature vector

# In[15]:


cv = CountVectorizer(inputCol="features", outputCol="featurevector", minDF=2, vocabSize=2000000)


# In[16]:


model=cv.fit(featuredf)


# In[17]:


featurevec = model.transform(featuredf).drop("features")


# In[18]:


featurevec = featurevec.withColumnRenamed("featurevector", "features")


# In[19]:


featurevec.show(5)


# ## Build LSH 

# Eliminate any zero length feature vectors from the input

# In[20]:


vectorlength = f.udf(lambda x: x.numNonzeros(), IntegerType())


# In[21]:


arraylength = f.udf(lambda x: len(x), IntegerType())


# In[22]:


f2 = featurevec.withColumn("vlen", vectorlength(featurevec.features))


# In[23]:


sparsevec = f2.where(f2.vlen > 1)


# In[24]:


mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)


# In[25]:


mhmodel = mh.fit(sparsevec)


# In[26]:


transform = mhmodel.transform(sparsevec)


# In[27]:


transform.show(5)


# In[28]:


transform.count()


# In[29]:


t2 = transform.withColumn("hlen", arraylength("hashes"))


# In[30]:


t2.orderBy("vlen").show(5)


# In[31]:


t2.orderBy("hlen").show(5)


# ## Load Challenge set

# In[32]:


mpd_test=spark.read.json("../mpd-challenge/challenge_set.json", multiLine=True)


# In[33]:


cpl=mpd_test.select(f.explode("playlists").alias("playlist"))


# In[34]:


recdf=cpl.select("playlist.name", "playlist.num_holdouts", "playlist.pid", "playlist.num_tracks", "playlist.tracks", "playlist.num_samples")


# In[35]:


chtracks = recdf.select("pid", "tracks.track_uri")


# In[36]:


chartist = recdf.select("pid", "tracks.artist_uri")


# In[37]:


challengedf = chtracks.join(chartist, chtracks.pid == chartist.pid).drop(chartist.pid)


# In[38]:


challengedf = challengedf.withColumn("features", mergeCols(f.col("track_uri"),f.col("artist_uri"))).drop("track_uri").drop("artist_uri")


# In[39]:


tokedf = mpd.canonicaltokens(recdf.select("pid", "name").fillna({"name": ""}), "name", "filtered").drop("name")


# In[40]:


challengedf.printSchema()


# In[41]:


tokedf.printSchema()


# In[42]:


challengedf = challengedf.join(tokedf, tokedf.pid == challengedf.pid).drop(tokedf.pid)


# In[43]:


challengedf.printSchema()


# In[44]:


challengedf = challengedf.withColumn("featurevec", mergeCols("filtered", "features")).drop("filtered").drop("features")


# In[45]:


challengedf = challengedf.withColumnRenamed("featurevec", "features")


# In[46]:


challengedf.printSchema()


# In[47]:


challengedf.show(5)


# ## Map challenge set into training vocab

# In[48]:


challengevec = model.transform(challengedf).drop("features")


# In[49]:


challengevec = challengevec.withColumnRenamed("featurevector", "features")


# In[50]:


challengevec.printSchema()


# In[51]:


challengevec.show(5)


# In[52]:


c2 = challengevec.withColumn("vlen", vectorlength("features"))


# In[53]:


c2.orderBy("vlen").show(5)


# There are challenge set vectors that have a length of zero.  This shouldn't cause any problems because can just recommend top songs from the global data set here.

# In[54]:


c2.where(c2.vlen == 0).describe("pid").show()


# ## Find playlist matches for one challenge set

# Select the first challenge playlist to test.`

# In[55]:


challengevec.cache()


# In[56]:


testpl = challengevec.limit(1)


# In[57]:


testpl.show(truncate=False)


# In[58]:


testvec=challengevec.select("features").rdd.map(lambda x: x.features).take(1)[0]


# In[59]:


testvec


# In[60]:


type(testvec)


# In[61]:


testpid=challengevec.select("pid").rdd.map(lambda x: x.pid).take(1)[0]


# In[62]:


testpid


# In[63]:


mh.getOutputCol()


# In[64]:


mhmodel.params


# In[65]:


transform.printSchema()


# In[66]:


transform.cache()


# In[67]:


hot100 = mhmodel.approxNearestNeighbors(transform, testvec, 100)


# In[68]:


# hot100.explain()


# In[69]:


hot100.count()


# In[70]:


hot100.printSchema()


# In[71]:


hot100.show(5)


# In[72]:


pidlist = hot100.select("pid").toPandas()["pid"].tolist()


# In[73]:


pidlist[0:5]


# In[74]:


type(pidlist)


# In[75]:


recommend = { "pid": testpid, "recpl": [pidlist]}


# In[76]:


recommend


# In[77]:


testpd = pd.DataFrame({"pid":0, "recpl":[]})


# In[78]:


testpd


# In[79]:


testpd.append(pd.DataFrame(recommend))


# In[80]:


recpd = pd.DataFrame(recommend)


# In[81]:


recpd


# In[82]:


recpd.append(pd.DataFrame(recommend))


# In[83]:


spark.createDataFrame(pd.DataFrame(recommend)).show()


# ## Build nearest neighbor playlists

# In[84]:


def getrecommend(chpl, model, transform):
    #testvec=challenge.select("features").rdd.map(lambda x: x.features).take(1)[0]
    #testpid=challenge.select("pid").rdd.map(lambda x: x.pid).take(1)[0]
    
    testvec = chpl.features
    testpid = chpl.pid
    #print("DEBUG: " + testvec + " " + testpid )
    
    hot100 = model.approxNearestNeighbors(transform, testvec, 100)
    
    pidlist = hot100.select("pid").toPandas()["pid"].tolist()
    
    recommend = { "pid": testpid, "recpl": [pidlist]}
    #print("DEBUG: " + testpid + " " + pidlist)
    
    return recommend
    #return recpd
    #print(requests.pid)
    #testvec = requests.features
    
    #return testpid, testvec


# In[85]:


testpd = pd.DataFrame({"pid":0, "recpl":[]})


# In[ ]:


for row in challengevec.rdd.collect():
    rec = getrecommend(row, mhmodel, transform)
    testpd = testpd.append(pd.DataFrame(rec))


# In[87]:


testpd


# In[88]:


testpd.to_pickle("neighborpl.pkl")


# In[89]:


testpd.to_csv("neighborpl.csv")

