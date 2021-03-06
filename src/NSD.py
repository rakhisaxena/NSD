import time
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import *
import scipy.stats as st
import sys
from os import listdir
from os.path import isfile, join
from igraph import *
import scipy.spatial.distance
import matplotlib.pyplot as plt
import csv
from scipy.stats import entropy
from scipy.stats import linregress
from numpy.linalg import norm
import pandas as pd
from mytruss import *

from utils import *

is_debug = False
numquartiles = 100

# import local config: Set your local paths in dev_settings.py
DATA_URL=""
SAVE_URL=""
try:
    from dev_settings import *
except ImportError:
    pass

def get_core_feature(g):
    cc = g.transitivity_local_undirected(mode=TRANSITIVITY_ZERO) 
    degree = g.degree()
       
    coreness = GraphBase.coreness(g)
    maxcore = max(coreness)
    #normalizedcoreness = [(i*1.0)/ maxcore for i in coreness]
    #edgediversity = getedgediversity(g)   #entropy of trussness of edges
    entropy = [0] * g.vcount()
    likeness = [0] * g.vcount()
    #sumnbrcores = [0] * g.vcount()
    #ccsamecore = [0] * g.vcount()
    #ccnotsamecore = [0] * g.vcount()
    
    for v in g.vs:
          ent = 0
          nbrcores = [coreness[w] for  w in g.neighbors(v)]
          likeness[v.index] = nbrcores.count(coreness[v.index]) #*1.0/degree[v.index]
                                 #len([w for w in nbrcores if w == coreness[v.index]] )  #*1.0/degree[v.index]

    feature = [(coreness[i.index],
                likeness[i.index],
                cc[i.index]               
              ) 
              for i in g.vs]
    return feature

        
def get_truss_feature(g):
    degree = g.degree()
    nodetruss = getnodetrussness(g)
    maxtruss = max(nodetruss)
    entropy = [0] * g.vcount()
    likeness = [0] * g.vcount()
    for v in g.vs:
          nbrtrusses = [nodetruss[w] for  w in g.neighbors(v)]
          if nbrtrusses :
             likeness[v.index] = nbrtrusses.count(nodetruss[v.index]) * 1.0/degree[v.index] 
    feature = [(nodetruss[i.index], 
                likeness[i.index]
              ) 
              for i in g.vs]
    return feature


def get_degree_feature(g): 
    degree = g.degree()
    likeness = [0] * g.vcount()
    for v in g.vs:
          nbrdegrees = [degree[w] for  w in g.neighbors(v)]
          likeness[v.index] = nbrdegrees.count(degree[v.index])*1.0/degree[v.index]

    feature = [(degree[i.index],
                likeness[i.index]              
              ) 
              for i in g.vs]
    return feature


def get_core_feature_faster(g):
    cc = g.transitivity_local_undirected(mode=TRANSITIVITY_ZERO) 
    degree = g.degree()
       
    coreness = GraphBase.coreness(g)
    maxcore = max(coreness)
    likeness = [0] * g.vcount()
    affinity = [0] * g.vcount()

    for e in g.es:        
      if (coreness[e.source] == coreness[e.target]) :
         likeness[e.source] += 1
         likeness[e.target] += 1

    affinity = [(x*1.0)/y for x, y in zip(likeness,degree)]
    feature = [(coreness[i.index],
                affinity[i.index],
                cc[i.index]               
              ) 
              for i in g.vs]
    return feature


def get_features(g,method):
    if (method.lower()=="core"):
       feature = get_core_feature_faster(g)
    if (method.lower()=="truss"):
       feature = get_truss_feature(g)
    if (method.lower()=="degree"):
       feature = get_degree_feature(g)
    
    #colorlist = {0:"red", 1:"orange", 2:"green",3:"yellow", 4:"pink",5:"blue", 6:"azure",7:"cyan",8:"magenta",9:"purple",10:"white" }
    #plot(g,vertex_label=[v.index for v in g.vs ],vertex_color=[colorlist[x] for x in coreness],layout=g.layout("kk"))
    return feature

def get_features_all(graphs):
    """
    Returns all features of all graphs.
    Out Format: {g1:[(f1..f7),(f1..f7),(f1..f7)...#nodes], g2:...}
    """
    # Order all the graphs names based on the timestamp
    #ordered_names = sorted(graphs.keys(), key=lambda k:int(k.split('_',1)[0]))
    #return {g: get_features(graphs[g], g) for g in ordered_names}
    return {g: get_features(graphs[g], g) for g in graphs}

def get_percentiles(feat):
    #print 'feat: ', feat
    feat_df = pd.DataFrame(feat) #zip(*feat)
    #print 'feat_df: ',feat_df
    signature = []
    for column in feat_df:
        #print 'column: ',feat_df[column].tolist()
        #print 'linspace: ', (np.linspace(0.0, 1.0, 9, 0))
        percentiles = feat_df[column].quantile(np.linspace(0, 1.0, numquartiles, 0)).tolist()    # 0.1, 1.0, 9, 0
        signature = signature +  percentiles 
        #print 'percentile : ',percentiles
    #print 'final signature: ', signature
    return signature
    
    
def aggregator(features_all):
    #print("Aggregating features")
    return {g: get_moments(features_all[g]) for g in features_all}

def canberra_dist(sig1, sig2):
    """
    Returns the Canberra distance between graphs described by the signatures
    sig1 and sig2.
    """
    return abs(scipy.spatial.distance.canberra(sig1, sig2))

def saveDists(graph_names, dists, file_name):
    #assert (len(graph_names) - len(dists) == 1),\"len(graph_names) - len(dists) != 1"
    #print("graph names")
    #print(graph_names)
    data = zip(graph_names[0:],dists)
    #print("data: ", data)
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(graph_names)
        writer.writerows(data)
        #writer.writerows(dists)

def saveDict(datadict, file_name):
    data = zip(datadict.keys(),datadict.values())
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sorted(data))
    return

def saveFeature(key, values, file_name):
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([key] + values)
    return        

def saveFeatures(features, file_name):
    data = zip(features.keys(),features.values())
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sorted(data))
                      
def find_NS_distance(sigs):
    sigkeys= sorted(sigs.keys())
    items = len(sigs) 
    out_slow = np.ones((items,items))
    for j in xrange(0, items):
        for k in xrange(j, items):
            out_slow[j, k] = canberra_dist(sigs[sigkeys[j]], sigs[sigkeys[k]]) 
            out_slow[k, j] = out_slow[j, k]
    return out_slow

def compareNS(sigs):
        for g in sigs:
            assert (len(sigs[g])==7*5),"Total features != 7*5"
            
        dists = find_NS_distance(sigs)
        #print("distance matrix");
        #print(dists)
        #print('ordered_graphs = ', ordered_graphs)
        saveDists(sorted(sigs.keys()), dists, sys.argv[1]+"_ExtendedNSdists.txt")

def readDict(filename, sep):
    with open(filename, "r") as f:
        dict = {}
        for line in f:
            values = line.split(sep)
            dict[values[0]] = [float(x) for x in values[1:len(values)]]
        print dict
        return(dict)

def distance(sigs) :
    sigkeys= sorted(sigs.keys())
    l = len(sigkeys) 
    DistanceMatrix = pd.DataFrame(np.ones((l,l)), index=sigkeys, columns=sigkeys)
    for i in range(l):
       g1name = sigkeys[i]
       for j in range(i,l):
            g2name = sigkeys[j]
            #DistanceMatrix[g1name][g2name] = LineRegress_dist(sigs[g1name], sigs[g2name]) 
            DistanceMatrix[g1name][g2name] = Canberra_dist(sigs[g1name], sigs[g2name]) 
            DistanceMatrix[g2name][g1name] = DistanceMatrix[g1name][g2name]
      
    return DistanceMatrix

    
def distanceFromFile():  
    inputdirectory = sys.argv[1]
    method = sys.argv[2]
    graphclass = sys.argv[1].split('/')[-1].split('.')[0]
    print 'graphclass: ', graphclass
    directory = outputdirectory+ graphclass + "/"
    print 'directory: ', directory
    if not os.path.exists(directory):
           os.makedirs(directory) 
    signaturesFile = directory+graphclass+"_Quantile"+method+"NetSimileSignatures.txt"    
    sigs = readDict(signaturesFile, ',')
    print("Read signaturesFile")
    sigkeys= sorted(sigs.keys())
    l = len(sigkeys) 
    DistanceMatrix = pd.DataFrame(np.ones((l,l)), index=sigkeys, columns=sigkeys)
    for i in range(l):
       g1name = sigkeys[i]
       for j in range(i,l):
            g2name = sigkeys[j]
            #DistanceMatrix[g1name][g2name] = LineRegress_dist(sigs[g1name], sigs[g2name])  
            DistanceMatrix[g1name][g2name] = EarthMovers_dist(sigs[g1name], sigs[g2name]) 
            DistanceMatrix[g2name][g1name] = DistanceMatrix[g1name][g2name]
  
    DistanceMatrix.to_csv(directory+graphclass+"_Quantile"+method+"NetSimileDists-EarthMovers.txt") 
    print 'OutputFile: ', directory+graphclass+"_Quantile"+method+"NetSimileDists-EarthMovers.txt"
    return 

def distanceFromFileSingleFeature():  
    #inputdirectory = sys.argv[1]
    method = sys.argv[2]
    graphclass = sys.argv[1].split('/')[-1].split('.')[0]
    print 'graphclass: ', graphclass
    directory = outputdirectory+ graphclass + "/"
    print 'directory: ', directory
    if not os.path.exists(directory):
           os.makedirs(directory) 
    signaturesFile = directory+graphclass+"_Feature_"+method+".txt"    
    sigs = readDict(signaturesFile, ',')
    print("Read signaturesFile")
    sigkeys= sorted(sigs.keys())
    l = len(sigkeys) 
    DistanceMatrix = pd.DataFrame(np.ones((l,l)), index=sigkeys, columns=sigkeys)
    for i in range(l):
       g1name = sigkeys[i]
       for j in range(i,l):
            g2name = sigkeys[j]
            #DistanceMatrix[g1name][g2name] = LineRegress_dist(sigs[g1name], sigs[g2name])  
            DistanceMatrix[g1name][g2name] = Canberra_dist(sigs[g1name], sigs[g2name]) 
            DistanceMatrix[g2name][g1name] = DistanceMatrix[g1name][g2name]
  
    DistanceMatrix.to_csv(directory+graphclass+"_Feature_"+method+"Dists.txt") 
    print 'OutputFile: ', directory+graphclass+"_Feature_"+method+"Dists.txt"
    return 
#==============================================================================
# NSD algorithm
#==============================================================================
def getNSSignatures():
   
   inputdirectory = sys.argv[1]
   method = sys.argv[2].lower()
   if method not in ('core','truss','degree'):
       print 'Incorrect method (argv2)'
       return
   
   graphclass = sys.argv[1].split('/')[-1].split('.')[0]
   print 'graphclass: ', graphclass
   directory = outputdirectory+ graphclass + "/"
   print 'directory: ', directory
   if not os.path.exists(directory):
           os.makedirs(directory) 
   nsoutputfile = directory+graphclass+"_Quantile"+method+"NetSimileSignatures.txt"
   nstimingfile = directory+graphclass+"_Quantile"+method+"NetSimileTimings.txt"
   nsdistsfile = directory+graphclass+"_Quantile"+method+"NetSimileDists.txt"
   
   os.remove(nsoutputfile) if os.path.exists(nsoutputfile) else None
   os.remove(nstimingfile) if os.path.exists(nstimingfile) else None
   
   print 'nsoutputfile: ', nsoutputfile
   NetSimileSignatures = {}
   nsTimings = {}
   for file in listdir(sys.argv[1]):
       graphfile = inputdirectory+'/'+file
       print graphfile
       start_time = time.time()
       graphname = graphfile.split('/')[-1].split('.')[0]
       graph = Graph.Read_Ncol(graphfile, directed=False,weights=False)
       graph = graph.simplify()
       graph.vs.select(_degree = 0).delete()
       n = graph.vcount()
       e = graph.ecount()
       print 'After reading num edges : ', e
       print ("Extracting NCKD ",method," features: %s" % graphname) 
       features = get_features(graph,method)
       NetSimileSignature = get_percentiles(features) 
       #NetSimileSignature =get_moments(features)
       NetSimileSignatures[graphname] = NetSimileSignature      
       nsTimings[graphname] = time.time() - start_time
       saveFeature(graphname,NetSimileSignature, nsoutputfile)
       
   saveDict(nsTimings, nstimingfile)
   #dists = find_NS_distance(NetSimileSignatures)
   #saveDists(sorted(NetSimileSignatures.keys()), dists, nsdistsfile)
   
   DistanceMatrix = distance(NetSimileSignatures)
   DistanceMatrix.to_csv(nsdistsfile) 
   return


#==============================================================================
# Main
# Command line parameter: name-of-dataset 
#==============================================================================

outputdirectory = '../NCOutput/'   
if __name__=="__main__":
  getNSSignatures()        
  #distanceFromFile()
  #distanceFromFileSingleFeature()#
