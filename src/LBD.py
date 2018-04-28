import time
from cv2 import *
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

#from utils import *

is_debug = False

# import local config: Set your local paths in dev_settings.py
DATA_URL=""
SAVE_URL=""
try:
    from dev_settings import *
except ImportError:
    pass
    
    
def get_subgraph(graph, node, radius=2):
    neighbors = graph.neighborhood(node, order= radius)  
    return graph.induced_subgraph(neighbors)

def get_li(graph):  
    m = graph.maxdegree(graph.vs)
    l = 0
    for v in graph.vs:
       l += (m - graph.degree(v))
    n = graph.vcount() * 1.0
    leadership = 0.0
    if (n > 3):
       leadership = l /((n-2)*(n-1))
    return leadership
 
def get_bi(graph):
    return graph.transitivity_undirected()

def get_di(graph):
    disjoint_dipoles = 0
    for e1 in graph.es:
      for e2 in graph.es:
         if (not(graph.are_connected(e1.source,e2.source)) and not(graph.are_connected(e1.source,e2.target)) and not(graph.are_connected(e1.target,e2.target)) and not(graph.are_connected(e1.target,e2.source))):
            disjoint_dipoles +=1;
    disjoint_dipoles /= 2                        # because each disjoint pair is counted twice
    n = graph.vcount() * 1.0
    diversity = 0.0
    if (n >= 4):
       divisor = (n/4)*((n/2)-1) * (n/4)*((n/2)-1)  
       diversity = math.sqrt(disjoint_dipoles / divisor)
    return diversity
    
         
def get_feature(name, graph):
    print ("Extracting LBD features: %s" % name) 
    l = []
    b = []
    d = []
    for i in graph.vs:
       subgraph = get_subgraph(graph, i)
       l.append(get_li(subgraph))
       b.append(get_bi(subgraph))
       d.append(get_di(subgraph))
    return l,b,d

def saveDict(datadict, file_name):
    data = zip(datadict.keys(),datadict.values())
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sorted(data))
    return
    
def saveDists(graph_names, dists, file_name):
    data = zip(graph_names[0:],dists)
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(graph_names)
        writer.writerows(data)
    
def saveFeature(key, values, file_name):
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([key] + values)
    return



def emd(s1,s2):
  x = np.array(s1)
  y = np.array(s2)
  #print("x = ", x ," y = " , y) 
  a = np.zeros((len(x),2))
  b = np.zeros((len(y),2))
  for i in range(0,len(a)):
    a[i][0] = x[i]
    a[i][1] = i+1.0

  for i in range(0,len(b)):
    b[i][0] = y[i]
    b[i][1] = i+1.0
  #print("a = ", a ," b = " , b) 
  # Convert from numpy array to CV_32FC1 Mat
  a64 = cv.fromarray(a)
  a32 = cv.CreateMat(a64.rows, a64.cols, cv.CV_32FC1)
  cv.Convert(a64, a32)

  b64 = cv.fromarray(b)
  b32 = cv.CreateMat(b64.rows, b64.cols, cv.CV_32FC1)
  cv.Convert(b64, b32)

  # Calculate Earth Mover's
  dis = cv.CalcEMD2(a32,b32,cv.CV_DIST_L2)    #CV_DIST_L2 -- Euclidean Distance, CV_DIST_L1 --- Manhattan Distance 
  return dis
  
def find_EMD_distance(sigs):
    sigkeys= sorted(sigs.keys())
    items = len(sigs) 
    out_slow = np.ones((items,items))
    for j in xrange(0, items):
        for k in xrange(j, items):
            a = np.array(sigs[sigkeys[j]])
            b = np.array(sigs[sigkeys[k]])
            out_slow[j, k] = emd(a, b) 
            out_slow[k, j] = out_slow[j, k]
    return out_slow

def readDict(filename, sep):
    with open(filename, "r") as f:
        dict = {}
        for line in f:
            values = line.split(sep)
            dict[values[0]] = [float(x) for x in values[1:len(values)]]
        #print dict
        return(dict)
           

     
def compareLBDfromFile(filename):
    sigs = readDict(filename, ',')
    sigkeys= sigs.keys()
    l = len(sigkeys) 
    distanceMatrix = pd.DataFrame(np.ones((l,l)), index=sigkeys, columns=sigkeys)
    for i in range(l):
       g1name = sigkeys[i]
       for j in range(i,l):
            g2name = sigkeys[j]
            a = np.array(sigs[g1name])
            b = np.array(sigs[g2name])
            distanceMatrix[g1name][g2name] = emd(a,b) 
            distanceMatrix[g2name][g1name] = distanceMatrix[g1name][g2name]
    distanceMatrix.to_csv(sys.argv[1].split('_')[0]+"_LBDdists.txt")   
    return distanceMatrix
 

def newcompareLBDfromFile(filename):
    sigs = readDict(filename, ',')
    sigkeys= sigs.keys()
    l = len(sigkeys) 
    distanceMatrix = pd.DataFrame(np.ones((l,l)), index=sigkeys, columns=sigkeys)
    for i in range(l):
       g1name = sigkeys[i]
       l1 = sigs[g1name][0:5]
       #print g1name, ' s1: ', sigs[g1name]
       #print '#l1',l1
       b1 = sigs[g1name][5:10]
       d1 = sigs[g1name][10:15]
       
       for j in range(i,l):
            g2name = sigkeys[j]
            
            l2 = sigs[g2name][0:5]
            #print g2name, ' s2: ', sigs[g2name]
            #print '*l2',l2
            b2 = sigs[g2name][5:10]
            d2 = sigs[g2name][10:15]
            
            ldis = emd(l1,l2) 
            bdis = emd(b1,b2)
            ddis = emd(d1,d2)
            
            meandis = (ldis + bdis + ddis)/3
            #print 'ldis',ldis,'bdis',bdis,'ddis',ddis,'meandis', meandis
            
            distanceMatrix[g1name][g2name] =  meandis
            distanceMatrix[g2name][g1name] = distanceMatrix[g1name][g2name]
    distanceMatrix.to_csv(sys.argv[1].split('_')[0]+"_LBDdists.txt")   
    return distanceMatrix
    
    
#==============================================================================
# LBD algorithm
#==============================================================================
def LBD(graph_files, dir_path):
    LBDTimings = {}
    for g in graph_files:
       #print('graph: ', g)
       graph = Graph.Read_Ncol(join(dir_path, g), directed=False)
       #print("NumVertices=", graph.vcount(), " NumEdges=", graph.ecount())
       start_time = time.time()
       bins = [0,0.2,0.4,0.6,0.8,1.0]
       l,b,d = get_feature(g, graph)
       lhist = (np.histogram(l, bins = bins)[0] / (graph.vcount()*1.0)).tolist()
       bhist = (np.histogram(b, bins = bins)[0] / (graph.vcount()*1.0)).tolist()
       dhist = (np.histogram(d, bins = bins)[0]/ (graph.vcount()*1.0)).tolist()
       LBDSignature = [lhist , bhist , dhist ]
       #print('LBD Signature: ', LBDSignature)
       LBDTimings[g] = [time.time() - start_time]
       saveFeature(g, LBDSignature, sys.argv[1]+"_LBDSignatures.txt")
       saveFeature(g, LBDTimings[g], sys.argv[1]+"_LBDTimings.txt")
    return


def LBD_fullGraph(graph_files, dir_path):
   for g in graph_files:
       graph = Graph.Read_Ncol(join(dir_path, g), directed=False)
       print('Graph: ', g, " NumVertices=", graph.vcount(), " NumEdges=", graph.ecount())
       print('L B D = ', get_li(graph), get_bi(graph), get_di(graph))
   return
   

#==============================================================================
# Main
# Command line parameter: name-of-dataset
# Example Usage:$ python LBD.py "reality_mining_voices"
#==============================================================================
if __name__=="__main__":
    #dir_path = join(DATA_URL, sys.argv[1])
    #print(dir_path)
    #graph_files = [f for f in listdir(dir_path) if \
     #                   isfile(join(dir_path,f)) ]
    #print("Files in Directory: ",graph_files)
    #LBD(graph_files, dir_path)
    #LBD_fullGraph(graph_files, dir_path)
    newcompareLBDfromFile(sys.argv[1])
    #filename = sys.argv[1] + "NS-Data_LBDSignatures.txt"
    #print filename
    #newcompareLBDfromFile(filename)
    
    #a = [0.835, 0.163, 0.001, 0.0, 0.001]
    #b = [0.198170732, 0.530487805, 0.175813008, 0.029471545, 0.066056911]
    #print 'emd', emd(a,b)
    
