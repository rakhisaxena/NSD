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
from numpy.linalg import norm
import pandas as pd

from utils import *

is_debug = False

# import local config: Set your local paths in dev_settings.py
DATA_URL=""
SAVE_URL=""
try:
    from dev_settings import *
except ImportError:
    pass


#==============================================================================
# 7 feature functions and their helper functions
# Page 2: Berlingerio, Michele, et al. "NetSimile: a scalable approach to
# size-independent network similarity." arXiv preprint arXiv:1209.2684 (2012).
#==============================================================================
def get_egonet(node, graph):
    """
    A nodes egonet is the induced subgraph formed by the node and its
    neighbors. Returns list of vertices that belong to the egonet
    """
    return graph.neighborhood(node)

def get_di(node, graph):
    """
    Number of neigbors
    """
    return graph.neighborhood_size(node)

def get_ci(node, graph):
    """
    Clustering coefficient of node, defined as the number of triangles
    connected to node over the number of connected triples centered on node
    """
    return graph.transitivity_local_undirected(node, mode=TRANSITIVITY_ZERO)

def get_dni(node, graph):
    """
    Average number of nodes two-hop away neighbors
    """
    return mean([get_di(n, graph) for n in get_egonet(node, graph) \
                                  if n != node])

def get_cni(node, graph):
    """
    Average clustering coefficient of neighbors of node
    """
    return mean([get_ci(n, graph) for n in get_egonet(node, graph) \
                                  if n != node])

def get_eegoi(node, graph):
    """
    Number of edges in nodes egonet;
    """
    edges_to=[]
    vertices = get_egonet(node, graph)

    for n in vertices:
        for i in get_egonet(n, graph):
            if i!=n:
                edges_to.append(i)

    #remove external nodes
    edges2x=[i for i in edges_to if i in vertices]
    assert (len(edges2x)%2==0),"Wrong calculation"
    return len(edges2x)/2


def get_eoegoi(node, graph):
    """
    Number of outgoing edges from node's egonet
    """
    edges_to=[]
    vertices=get_egonet(node, graph)

    for n in vertices:
        for i in get_egonet(n, graph):
            edges_to.append(i)

    return len([i for i in edges_to if i not in vertices])

def get_negoi(node, graph):
    """
    Number of neighbors of node's egonet
    """
    vertices = get_egonet(node, graph)
    all_neighbors = []
    for v in vertices:
        all_neighbors = all_neighbors + get_egonet(v,graph)
    all_neighbors = set(all_neighbors)
    all_neighbors =  [i for i in all_neighbors if i not in vertices]
    return(len(all_neighbors))

#==============================================================================
# NetSimile Algorithm components
# Features: Number of neigbors
#	    Clustering coefficient of node
#	    Average number of nodes two-hop away neighbors
# 	    Average clustering coefficient of neighbors of node
#	    Number of edges in nodes egonet
#	    Number of outgoing edges from node's egonet
#	    Number of neighbors of node's egonet
#==============================================================================
def get_features(g):
    feature = [(get_di(i,g),
             get_ci(i,g),
             get_dni(i,g),
             get_cni(i,g),
             get_eegoi(i,g),
             get_eoegoi(i,g),
             get_negoi(i,g)) for i in g.vs]
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

def get_moments(feat):
    """
    input: feature matrix of a single graph
    output: for each feature, return the 5 moments
    """
    #print("features: ", feat)
    feat_cols = zip(*feat)
    assert (len(feat_cols)==7),"Total columns != 7"

    # Calculate the 5 aggregates for each feature
    signature = []
    for f in feat_cols:
        #print f
        signature = signature + [mean(f),
             median(f),
             std(f),
             st.skew(f),
             st.kurtosis(f)]
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
        saveDists(sorted(sigs.keys()), dists, sys.argv[1]+"_NSdists.txt")

def readDict(filename, sep):
    with open(filename, "r") as f:
        dict = {}
        for line in f:
            values = line.split(sep)
            dict[values[0]] = [float(x) for x in values[1:len(values)]]
        print dict
        return(dict)
        
#==============================================================================
# NetSimile algorithm
#==============================================================================
def NetSimile(graph_files, dir_path, use_old_dists=False):
    
    NetSimileSignatures = {}
    NetSimileTimings = {}

    for g in graph_files:
       graph = Graph.Read_Ncol(join(dir_path, g), directed=False)
       start_time = time.time()
       features = get_features(graph, g)
       NetSimileSignature = get_moments(features)
       NetSimileSignatures[g] = NetSimileSignature      
       NetSimileTimings[g] = time.time() - start_time
       saveFeature(g, NetSimileSignature, sys.argv[1]+"_NetSimileSignatures.txt")
    saveFeatures(NetSimileTimings, sys.argv[1]+"_NetSimileTimings.txt")
    dists = find_NS_distance(NetSimileSignatures)
    saveDists(sorted(NetSimileSignatures.keys()), dists, sys.argv[1]+"_NSdists.txt")
    return

def getNSSignatures():
   inputdirectory = sys.argv[1]
   graphclass = sys.argv[1].split('/')[-1].split('.')[0]
   print 'graphclass: ', graphclass
   directory = outputdirectory+ graphclass + "/"
   print 'directory: ', directory
   if not os.path.exists(directory):
           os.makedirs(directory) 
   nsoutputfile = directory+graphclass+"_NetSimileSignatures.txt"
   nstimingfile = directory+graphclass+"_NetSimileTimings.txt"
   nsdistsfile = directory+graphclass+"_NetSimileDists.txt"
   
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
       graph = Graph.Read_Ncol(graphfile, directed=False,weights=True)
       graph = graph.simplify()
       n = graph.vcount()
       e = graph.ecount()
       print 'After reading num edges : ', e
       print ("Extracting NetSimile features: %s" % graphname) 
       features = get_features(graph)
       NetSimileSignature = get_moments(features)
       NetSimileSignatures[graphname] = NetSimileSignature     
       endtime = time.time() - start_time 
       nsTimings[graphname] = endtime 
       saveFeature(graphname,NetSimileSignature, nsoutputfile)
       saveFeature(graphname,[endtime], nstimingfile)
       
  # saveDict(nsTimings, nstimingfile)
   #dists = find_NS_distance(NetSimileSignatures)
   #saveDists(sorted(NetSimileSignatures.keys()), dists, nsdistsfile)
   
   #CanberraDistanceMatrix = distance(NetSimileSignatures)
   #CanberraDistanceMatrix.to_csv(nsdistsfile) 
   return

def distanceFromFile():  
    inputdirectory = sys.argv[1]
    graphclass = sys.argv[1].split('/')[-1].split('.')[0]
    print 'graphclass: ', graphclass
    directory = outputdirectory+ graphclass + "/"
    print 'directory: ', directory
    if not os.path.exists(directory):
           os.makedirs(directory) 
    signaturesFile = directory+graphclass+"_NetSimileSignatures.txt"    
    sigs = readDict(signaturesFile, ',')
    print("Read signaturesFile")
    sigkeys= sorted(sigs.keys())
    l = len(sigkeys) 
    CanberraDistanceMatrix = pd.DataFrame(np.ones((l,l)), index=sigkeys, columns=sigkeys)
    for i in range(l):
       g1name = sigkeys[i]
       for j in range(i,l):
            g2name = sigkeys[j]
            CanberraDistanceMatrix[g1name][g2name] = Canberra_dist(sigs[g1name], sigs[g2name]) 
            CanberraDistanceMatrix[g2name][g1name] = CanberraDistanceMatrix[g1name][g2name]
  
    CanberraDistanceMatrix.to_csv(directory+graphclass+"_NetSimileDists.txt") 
    print 'OutputFile: ', directory+graphclass+"_NetSimileDists.txt"
    return 
  


#==============================================================================
# Main
# Command line parameter: name-of-dataset#
# Example Usage:$ python netsimile.py "reality_mining_voices"
#==============================================================================

outputdirectory = '../NCOutput/'   
if __name__=="__main__":
  #getNSSignatures()        
  distanceFromFile()
