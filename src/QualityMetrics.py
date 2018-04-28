import numpy as np
import warnings
import sklearn
import math
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report,confusion_matrix

def qualitymetrics(cmat):
 maxi = cmat.max(axis=0)		#cmat: confusion_matrix
 numpoints = sum(cmat.sum(0))
 
 nc2 = lambda n : n * (n-1) /2
 
 TPplusFPplusTNplusFN = nc2(numpoints)
 
 sumcols = cmat.sum(0)
 #print "sumcols: ", sumcols
 TPplusFP = sum(nc2(sumcols))
 TP = 0
 for x in range(cmat.shape[1]): 
    TP += sum(nc2(cmat[:,x]))
 
 TPplusFN = 0
 for x in range(cmat.shape[0]): 
    TPplusFN += nc2(sum(cmat[x,:]))
 
 FN = TPplusFN - TP
 TNplusFN = TPplusFPplusTNplusFN - TPplusFP
 TN = TNplusFN - FN
 
 #print "TPlusFP: ", TPplusFP
 #print "TP: ", TP
 #print "TPplusFN: ", TPplusFN
 #print "FN: ", FN
 #print "TNplusFN", TNplusFN
 #print "TN : ", TN
 
 n_classes = cmat.sum(axis=1) # Total number of data for each true label Class (each row)
 n_clusters = cmat.sum(axis=0) # Total number of data for each cluster (each col)
 rows = cmat.shape[1]
 cols = cmat.shape[0]
 normalized_n_classes = np.asarray([(float(x)/numpoints) for x in n_classes])
 normalized_n_clusters = np.asarray([(float(x)/numpoints) for x in n_clusters])
 log_n_classes = [ math.log(float(x) /numpoints) for x in n_classes ]
 log_n_clusters = [ math.log(float(x) /numpoints) for x in n_clusters ]
 entropy_classes = (-1) * sum(normalized_n_classes  * log_n_classes)
 entropy_clusters = (-1) * sum(normalized_n_clusters * log_n_clusters)
 
 mi = 0		#mutual information
 for k in range(cols): 
   cluster = cmat[:,k]
   #print "cluster", cluster
   wk = n_clusters[k]
   #print "wk", wk
   for j in range(rows):
    cj = n_classes[j]
    #print "cj", cj
    #print "cluster[j]", cluster[j]
    if not(cluster[j] == 0):
       mi +=   float(cluster[j])/numpoints  * math.log( float(numpoints) * cluster[j]/(wk*cj))
  

 nmi = 2 * mi / (entropy_classes + entropy_clusters)
 
 #print "numpoints",numpoints
 #print "n_classes",n_classes
 #print "normalized_n_classes",normalized_n_classes
 #print "log_n_classes", log_n_classes
 #print "product: ", normalized_n_classes  * log_n_classes
 #print "sum-product", sum(normalized_n_classes  * log_n_classes)
 #print "----"
 #print "n_clusters", n_clusters
 #print "normalized_n_clusters", normalized_n_clusters
 #print "log_n_clusters", log_n_clusters
 #print "product: ", normalized_n_clusters  * log_n_clusters
 #print "sum-product", sum(normalized_n_clusters  * log_n_clusters)
 
 #print "mi", mi
 #print "entropy_classes",entropy_classes
 #print "entropy_clusters",entropy_clusters
 #print "nmi", nmi
 
 purity = float(sum(maxi))/numpoints
 precision = float(TP) / TPplusFP
 recall = float(TP) /TPplusFN
 accuracy = float(TP+TN)/TPplusFPplusTNplusFN
 q = np.asarray([purity, precision, recall, accuracy, nmi])
 return np.around(q, decimals=4)

def getMetrics(algorithm, classes,clusters):
 print algorithm
 print "classes", classes
 print "clusters", clusters
 print "purity,precision, recall, accuracy , nmi", qualitymetrics(confusion_matrix(classes,clusters))

def old_quality():
 classes = np.array([1,1,1,1,1,2,2,2,2,2,1,3,3,3,3,1,1])
 clusters = np.array([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3])
 getMetrics("NLP Stanford Example",classes,clusters)
 print "---------------------------------------------------------------------"
 
 
 # NetSimile 38 graphs 
 print "38 Dendrograms"
 print "A-1    A-2    A-3    A-4    A-5 B10K-2 B10K-1  B1K-1  B1K-2   CA-1   CA-2   CA-3   CA-4 "
 print "CA-5 E10K-1 E10K-2  E1K-1  E1K-2 F10K-1 F10K-2  F1K-1  F1K-2    M-1    M-2    M-3    M-4"
 print "M-5    M-6   WC-1   WC-2   WC-3 W20K-1 W20K-2  W2K-1  W2K-2   FW-1   FW-2   FW-3" 
 
 classes  = np.array([1,1,1,1,1,2,2,2,2,3,3,3,3,3,8,8,8,8,4,4,4,4,9,9,9,9,9,9,5,5,5,6,6,6,6,7,7,7])
 clusters = np.array([1,1,1,1,1,2,2,2,2,3,3,3,3,3,1,1,1,1,4,4,4,4,1,1,1,1,1,1,5,5,5,6,6,6,6,7,8,9])
 getMetrics("nNCKD", classes,clusters)

 classes  = np.array([1,1,1,1,1,2,2,2,2,3,3,3,3,3,8,8,8,8,4,4,4,4,9,9,9,9,9,9,5,5,5,6,6,6,6,7,7,7])
 clusters = np.array([1,1,1,1,1,2,2,1,1,3,4,5,3,5,6,6,7,7,4,4,4,4,4,5,4,5,5,4,9,4,2,9,9,9,9,8,8,8])
 getMetrics("NetSimile",classes,clusters)

 clusters = np.array([1,1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5,5,5,4,4,4,4,4,4,5,1,6,7,7,7,7,8,9,8])
 getMetrics("Only Edges NCKD",classes,clusters)
 
 clusters = np.array([1,1,1,1,1,2,2,2,2,3,3,3,3,3,1,1,1,1,4,4,4,4,1,1,1,1,1,1,5,6,6,7,7,7,7,8,9,8])
 getMetrics("Avg NCKD",classes,clusters)
 
 clusters = np.array([1,1,1,1,1,2,2,2,2,3,3,3,3,3,4,4,4,4,5,5,5,5,4,4,4,4,4,4,5,1,6,7,7,7,7,8,9,8 ])
 getMetrics("Max NCKD",classes,clusters) 
 
 clusters = np.array([1,1,1,1,1,2,2,2,2,3,3,3,3,3,1,1,1,1,4,4,4,4,1,1,1,1,1,1,5,6,6,7,7,7,7,8,9,8])
 getMetrics("Min NCKD",classes,clusters)
 
 clusters = np.array([1,1,1,1,1,2,2,2,2,3,3,3,3,3,1,1,1,1,4,4,4,4,1,1,1,1,1,1,5,6,6,7,7,7,7,8,9,8 ])
 getMetrics("Fuse NCKD",classes,clusters)
 
 print "---------------------------------------------------------------------"
 
 print "AE Dendrograms"
 print "A-AG A-AP A-MJ A-PF A-PH A-TH E-AT E-CE E-EN E-OS E-SC"  
 
 classes  = np.array([2,2,2,2,2,2,1,1,1,1,1])  
 clusters = np.array([1,1,1,1,1,1,2,1,2,2,1 ])
 getMetrics("eNCKD",classes,clusters)
 
 classes  = np.array([2,2,2,2,2,2,1,1,1,1,1])  
 clusters = np.array([1,2,1,2,2,1,2,1,1,2,1])
 getMetrics("NetSimile",classes,clusters)
 
 clusters = np.array([1,1,2,2,2,2,2,1,2,2,1])
 getMetrics("Only Nodes NCKD",classes,clusters) 
 
 clusters = np.array([1,1,1,1,1,1,2,1,2,2,1 ])
 getMetrics("Only Edges NCKD", classes,clusters) 
 
 clusters = np.array([1,1,1,1,1,1,2,1,2,2,1])
 getMetrics("Avg NCKD",classes,clusters)
 
 clusters = np.array([1,1,1,1,1,1,2,1,2,2,1 ])
 getMetrics("Max NCKD",classes,clusters)
 
 clusters = np.array([1,1,2,2,2,2,2,1,2,2,1])
 getMetrics("Min NCKD",classes,clusters)
 
 clusters = np.array([1,1,1,1,1,1,2,1,2,2,1])
 getMetrics("Fuse NCKD",classes,clusters)
 
 clusters = np.array([1,1,1,1,1,1,2,1,2,2,1])
 getMetrics("Normalized Max NCKD", classes,clusters)
 
  
 print "---------------------------------------------------------------------"
  
 print "Co-Authorship Dendrograms"
 print "CA-1 CA-2 CA-3 CA-4 CA-5 DY-1 DY-2 DY-3 DY-4 DY-5 DC-1 DC-2 DC-3 DC-4 DC-5"
   
 classes  = np.array([2,2,2,2,2,1,1,1,1,1,3,3,3,3,3])  
 clusters = np.array([1,2,2,1,2,3,3,3,3,3,3,3,3,3,3 ])
 getMetrics("eNCKD",classes,clusters)
 
 classes  = np.array([2,2,2,2,2,1,1,1,1,1,3,3,3,3,3])  
 clusters = np.array([1,2,2,1,2,3,3,2,2,2,3,3,3,3,3])
 getMetrics("NetSimile",classes,clusters)
 
 clusters = np.array([1,2,2,2,2,3,3,3,3,3,3,3,3,3,3 ])
 getMetrics("Only Nodes NCKD",classes,clusters) 
 
 clusters = np.array([1,2,2,1,2,3,3,3,3,3,3,3,3,3,3])
 getMetrics("Only Edges NCKD",classes,clusters)
 
 clusters = np.array([1,2,2,1,2,3,3,3,3,3,3,3,3,3,3])
 getMetrics("Avg NCKD",classes,clusters) 
 
 clusters = np.array([1,2,2,1,2,3,3,3,3,3,3,3,3,3,3 ])
 getMetrics("Max NCKD",classes,clusters)
 
 clusters = np.array([1,2,2,2,2,3,3,3,3,3,3,3,3,3,3 ])
 getMetrics("Min NCKD", classes,clusters)
 
 clusters = np.array([1,2,2,1,2,3,3,3,3,3,3,3,3,3,3 ])
 getMetrics("Fuse NCKD", classes,clusters)
 
 clusters = np.array([1,2,2,1,2,3,3,3,3,3,3,3,3,3,3])
 getMetrics("Normalized Max NCKD", classes,clusters)

 print "---------------------------------------------------------------------"
  
 print "LBD Dendrograms"
 print "BA-1 BA-2 ER-1 ER-2 FF-1 FF-2 WS-1 WS-2 CA-3 CA-5 FW-1 FW-2 FW-3  M-1  M-2  M-3  M-4  M-5  M-6 WC-1 WC-2 "
 classes  = np.array([1,1,2,2,3,3,4,4,5,5,6,6,6,7,7,7,7,7,7,8,8]) 
 #classes  = np.array([1,1,8,8,4,4,6,6,2,2,7,7,7,3,3,3,3,3,3,5,5]) 
 
 clusters = np.array([1,1,3,3,4,4,6,6,2,2,7,8,7,3,3,3,3,3,3,5,2])
 getMetrics("nCKD",classes,clusters)
 
 clusters = np.array([2,2,8,8,4,4,5,5,6,6,1,1,1,3,7,3,7,7, 3,5,4])
 getMetrics("NetSimile",classes,clusters)
 
 clusters = np.array([ 2,2,4,4,3,3,3,3,8,5,6,6,2,1,1,1,1,1,1,7,2])
 getMetrics("LBD",classes,clusters)
 
 print "---------------------------------------------------------------------"
 
 print "LBD (minus WC networks) Dendrograms"
 print "BA-1 BA-2 ER-1 ER-2 FF-1 FF-2 WS-1 WS-2 CA-3 CA-5 FW-1 FW-2 FW-3  M-1  M-2  M-3  M-4  M-5  M-6"
 classes  = np.array([1,1,2,2,3,3,4,4,5,5,6,6,6,7,7,7,7,7,7]) 
 
 clusters = np.array([3,3,2,2,5,5,1,1,4,4,7,6,7,2,2,2,2,2,2])
 getMetrics("nCKD",classes,clusters)  
  
 clusters = np.array([7,7,3,3,1,1,1,1,4,4,5,5,6,2,2,2,2,2,2 ])
 getMetrics("NetSimile",classes,clusters)
  
 clusters = np.array([7,7,3,3,5,5,1,1,2,2,6,6,6,4,2,4,2,2,4 ])
 getMetrics("LBD",classes,clusters)
 print "---------------------------------------------------------------------"
 
 print "LBD (minus ER networks) Dendrograms"
 print "BA-1 BA-2 FF-1 FF-2 WS-1 WS-2 CA-3 CA-5 FW-1 FW-2 FW-3  M-1  M-2  M-3  M-4  M-5  M-6 WC-1 WC-2"
 classes  = np.array([1,1,2,2,3,3,4,4,5,5,5,6,6,6,6,6,6,7,7 ]) 
 
 clusters = np.array([7,7,4,4,1,1,3,3,6,5,6,2,2,2,2,2,2,3,3])
 getMetrics("nCKD",classes,clusters) 
 
 clusters = np.array([7,7,3,3,1,1,4,4,6,6,6,5,2,5,2,2,5,1,3  ])
 getMetrics("NetSimile",classes,clusters)
 
 clusters = np.array([ 6,6,1,1,1,1,4,4,5,5,3,2,2,2,2,2,2,7,3  ])
 getMetrics("LBD",classes,clusters)
 
 print "---------------------------------------------------------------------"
 print "NCKD 32 (minus WC FW networks) Dendrograms"
 print "A-1    A-2    A-3    A-4    A-5 B10K-2 B10K-1  B1K-1  B1K-2   CA-1   CA-2   CA-3   CA-4 "
 print "CA-5 E10K-1 E10K-2  E1K-1  E1K-2 F10K-1 F10K-2  F1K-1  F1K-2    M-1    M-2    M-3    M-4"
 print "M-5    M-6  W20K-1 W20K-2  W2K-1  W2K-2 " 
 
 classes  = np.array([1,1,1,1,1,2,2,2,2,3,3,3,3,3,7,7,7,7,4,4,4,4,6,6,6,6,6,6,5,5,5,5])
 clusters = np.array([7,7,7,7,7,6,6,6,6,3,3,3,5,5,1,1,1,1,4,4,4,4,1,1,1,1,1,1,2,2,2,2])
 getMetrics("eNCKD", classes,clusters)
 
 clusters = np.array([6,6,6,6,6,6,6,7,7,5,4,2,5,2,1,1,3,3,4,4,4,4,4,2,4,2,2,4,2,2,2,2])
 getMetrics("Netsimile", classes,clusters)
 

 
 print "---------------------------------------------------------------------"
 print "LBD 14 (minus WC FW networks) Dendrograms"
 print "BA-1 BA-2 FF-1 FF-2 WS-1 WS-2 CA-3 CA-5  M-1  M-2  M-3  M-4  M-5  M-6"
 classes  = np.array([1,1,2,2,3,3,4,4,5,5,5,5,5,5]) 
 
 clusters = np.array([1,1,3,3,3,3,4,5,2,2,2,2,2,2])
 getMetrics("LBD", classes,clusters)
 
 clusters = np.array([1,1,5,5,3,3,4,4,2,4,2,4,4,2])
 getMetrics("NetSimile", classes,clusters)
 
 clusters = np.array([1,1,5,5,2,2,4,4,3,3,3,3,3,3])
 getMetrics("eNCKD", classes,clusters)
 return
 

def JNCKD_quality_old():
 print 'AS-1 AS-2 AS-3 AS-4 AS-5 CA-1 CA-2 CA-3 CA-4 CA-5 FB-1 FB-2 FB-3 FB-4 FB-5 FW-1 FW-2 FW-3 FW-4 FW-5 PP-1 PP-2 PP-3 PP-4 PP-5 SC-1 SC-2 SC-3 SC-4 SC-5 WT-1 WT-2 WT-3 WT-4 WT-5'
 classes =  np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7])
 
 print 'JNCKD Real NetEMD Dendro'
 clusters = np.array([1,1,1,1,1,2,2,1,1,2,3,3,3,3,3,4,4,3,3,3,3,4,4,2,2,5,6,3,3,3,7,7,7,7,7])
 getMetrics("JNCKD Real NetEMD",classes,clusters)
 print "---------------------------------------------------------------------"
 clusters = np.array([1,1,1,1,1,2,2,2,2,2,3,4,3,3,3,5,5,5,5,5,6,6,6,6,6,4,4,4,5,5,7,7,7,7,7])
 print 'JNCKD NetSimile Dendro'
 getMetrics("JNCKD NetSimile Real",classes,clusters)
 print "---------------------------------------------------------------------"
 clusters = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,4,4,7,7,7,7,7 ])
 print 'JNCKD Coreness-quartiles Dendro'
 getMetrics("JNCKD Real Coreness-quartiles",classes,clusters)
 print "---------------------------------------------------------------------"
 clusters = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,5,6,6,5,5,7,7,7,7,7 ])
 print 'JNCKD Trussness-quartiles Dendro'
 getMetrics("JNCKD Real Trussness-quartiles",classes,clusters)
 print "---------------------------------------------------------------------"
 

def JNCKD_quality():
 print 'AS-1 AS-2 AS-3 AS-4 AS-5 CA-1 CA-2 CA-3 CA-4 CA-5 FB-1 FB-2 FB-3 FB-4 FB-5 FW-1 FW-2 FW-3 FW-4 FW-5 ME-1 ME-2 ME-3 ME-4 ME-5 PP-1 PP-2 PP-3 PP-4 PP-5 SC-1 SC-2 SC-3 SC-4 SC-5 WT-1 WT-2 WT-3 WT-4 WT-5'
 classes =  np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8])
 
 print 'JNCKD Real NetEMD Dendro'
 clusters = np.array([1,1,1,1,1,1,1,1,1,1,2,1,2,2,2,2,2,3,3,2,4,4,4,4,4,2,1,1,1,1,1,5,5,6,7,8,8,8,8,8])
 getMetrics("JNCKD Real NetEMD",classes,clusters)
 print "---------------------------------------------------------------------"
 print 'JNCKD Real NetSimile Dendro'
 clusters = np.array([1,1,1,1,1,2,2,2,2,2,3,4,3,3,3,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,5,4,4,4,5,8,8,8,8,8 ])
 getMetrics("JNCKD Real NetSimile Real",classes,clusters)
 print "---------------------------------------------------------------------"
 print 'JNCKD Real NCKD Dendro'
 clusters = np.array([ 1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,4,5,5,5,5,6,6,7,8,8 ])
 getMetrics("JNCKD Real NCKD Real",classes,clusters)
 print "---------------------------------------------------------------------"
 print 'JNCKD Real Coreness-Quantiles Dendro'
 clusters = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8])
 getMetrics("JNCKD Real Coreness-quartiles",classes,clusters)
 print "---------------------------------------------------------------------"
 print 'JNCKD Trussness-Quantiles Dendro'
 clusters = np.array([1,1,1,1,1,2,2,3,2,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8])
 getMetrics("JNCKD Real Trussness-quartiles",classes,clusters)
 print "---------------------------------------------------------------------"
 print 'JNCKD Degree-Quantiles Dendro'
 clusters = np.array([1,1,1,1,1,2,2,3,2,3,4,4,4,4,4,5,5,5,5,5,3,3,3,3,3,1,1,1,1,1,6,7,7,7,7,8,8,8,8,8 ])
 getMetrics("JNCKD Real Degree-quartiles",classes,clusters)
 print "---------------------------------------------------------------------"
 
 print 'JNCKD Real Coreness-Quantiles-Cosine Dendro'
 clusters = np.array([ 1,1,1,1,1,2,2,3,3,2,4,5,4,5,4,6,6,6,6,6,7,7,7,7,7,4,4,4,4,4,7,7,7,7,8,6,6,6,6,6])
 getMetrics("JNCKD Real Coreness-quartiles Cosine",classes,clusters)
 print "---------------------------------------------------------------------"
 print 'JNCKD Trussness-Quantiles-Cosine Dendro'
 clusters = np.array([  1,1,1,1,1,2,2,3,4,1,2,2,2,2,2,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,5,7,7,8,7,5,5,5,5,5 ])
 getMetrics("JNCKD Real Trussness-quartiles Cosine",classes,clusters)
 print "---------------------------------------------------------------------"
 
 print 'JNCKD Real Coreness-Quantiles-Euclidean Dendro'
 clusters = np.array([1,1,1,1,1,1,1,1,2,1,3,4,3,4,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,6,6,5,6,7,7,7,8,8 ])
 getMetrics("JNCKD Real Coreness-quartiles Euclidean",classes,clusters)
 print "---------------------------------------------------------------------"
 print 'JNCKD Trussness-Quantiles-Euclidean Dendro'
 clusters = np.array([1,1,1,1,1,2,1,1,3,1,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,4,5,5,6,5,7,7,7,8,8 ])
 getMetrics("JNCKD Real Trussness-quartiles Euclidean",classes,clusters)
 print "---------------------------------------------------------------------"
 
 print 'JNCKD Real Coreness-Quantiles-EarthMovers Dendro'
 clusters = np.array([1,1,2,1,3,1,2,1,3,2,4,5,4,6,5,7,8,7,7,7,7,5,7,5,5,7,4,8,4,6,2,7,7,7,8,1,1,1,1,1])
 getMetrics("JNCKD Real Coreness-quartiles EarthMovers",classes,clusters)
 print "---------------------------------------------------------------------"
 print 'JNCKD Trussness-Quantiles-EarthMovers Dendro'
 clusters = np.array([1,1,1,2,3,1,1,2,3,2,4,5,4,5,4,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8,1,2,6,2,6,5,5,4,5,4 ])
 getMetrics("JNCKD Real Trussness-quartiles EarthMovers",classes,clusters)
 print "---------------------------------------------------------------------"
  
 
 
 print 'SN-1 SN-2 SN-3 SN-4 SN-5 RO-1 RO-2 RO-3 RO-4 RO-5 WB-1 WB-2 WB-3 WB-4 WB-5 '
 print 'JNCKD Big Trussness-quartiles'
 classes = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])
 clusters = np.array([1,1,1,1,1,2,2,2,2,2,3,2,2,3,3])
 getMetrics("JNCKD Big Trussness-quartiles",classes,clusters)
 print "---------------------------------------------------------------------"
 print 'JNCKD Big NetSimile Dendro'
 classes = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])
 clusters = np.array([1,1,1,1,1,2,2,2,2,2,3,3,2,2,2])
 getMetrics("JNCKD Big NetSimile",classes,clusters)
 print "---------------------------------------------------------------------"
 print 'JNCKD Big NCKD Dendro'
 classes = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])
 clusters = np.array([1,1,1,1,1,2,2,3,2,2,1,1,1,1,1 ])
 getMetrics("JNCKD Big NCKD",classes,clusters)
 print "---------------------------------------------------------------------"
  

if __name__=="__main__":
  warnings.filterwarnings("ignore")
  #quality()
  JNCKD_quality()



 
 #print "Confusion Matrix"
 #print confusion_matrix(classes,clusters)
 #print classification_report(classes,clusters)
