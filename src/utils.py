# -*- coding: utf-8 -*-
"""
@author: rakhi saxena
"""
import csv
from igraph import *
from os.path import join
import scipy as sp
import scipy.spatial.distance
#from cv2 import *
import numpy as np
import cv as cv
from scipy.stats import linregress
import numpy as np 
import pylab 
import scipy.stats as stats


#==============================================================================
# Utility functions
#==============================================================================
def file2igraph(file):
    """
    Converts graph file into iGraph object, adds artifacts
    """
    with open(file, 'r') as fi:
        v,e = fi.next().split()
        e_list = [(int(i.split()[0]), int(i.split()[1])) for i in list(fi)]
        assert (int(e) == len(e_list)),\
                    "#edges mentioned and # of edges in file differ"
        g = Graph()
        g.add_vertices(int(v))
        g.add_edges(e_list)
        return g

def saveDists(graph_names, dists, file_name):
    assert (len(graph_names) - len(dists) == 1),\
                "len(graph_names) - len(dists) != 1"
    data = zip(graph_names[1:],dists)
    with open(join("out",file_name), 'w') as fo:
        fr = csv.writer(fo)
        fr.writerows(data)

def loadDists(name):
    with open(join('out',name+"_dists.csv")) as fi:
        fr = csv.reader(fi, delimiter=',')
        data=list(fr)
    return [float(i[1]) for i in data]

def saveAnomalies(lim_up, lim_down, dists, anomalies):
    """
    Writes to disk the anomalies in the defined format.
    """
    with open(join('out',sys.argv[1]+'_anomalies.txt'), 'w') as fo:
        fo.write(str(lim_up)+' '+str(lim_down)+'\n')
        for a in anomalies:
            fo.write(str(a+1)+' '+str(dists[a]) +' ' + str(dists[a+1])+'\n')

def saveFeature(key, values, file_name):
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([key] + values)
    return

def saveDict(datadict, file_name):
    data = zip(datadict.keys(),datadict.values())
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sorted(data))
    return
        
def readDict(filename, sep):
    with open(filename, "r") as f:
        dict = {}
        for line in f:
            values = line.split(sep)
            dict[values[0]] = [float(x) for x in values[1:len(values)]]
        return(dict)
           

def Canberra_dist(sig1, sig2):
    """
    Returns the Canberra distance between graphs described by the signatures
    sig1 and sig2.
    """
    #print "In Canberra Distnace : "
    #print "Sig1 : ", sig1
    #print "Sig2 : ", sig2
    maxLen = max(len(sig1), len(sig2))
    sig1 = sig1 + [0]*(maxLen - len(sig1))
    sig2 = sig2 + [0]*(maxLen - len(sig2))
    d = abs(scipy.spatial.distance.canberra(sig1, sig2))
    #dis = 0
    #for i in range(maxLen) :
    #  denom = abs(sig1[i])+abs(sig2[i])
    #  if not(denom == 0) :
    #    dis = dis+ (abs(sig1[i] - sig2[i]) /denom)
    #print "Canberra distance: ", d
    #print "My Canberra distance: ", dis
    if (math.isnan(d)):
       d = 1.0
    return d #abs(scipy.spatial.distance.canberra(sig1, sig2))
    
def Euclidean_dist(sig1, sig2):
    """
    Returns the Euclidean distance between graphs described by the signatures
    sig1 and sig2.
    """
    maxLen = max(len(sig1), len(sig2))
    sig1 = sig1 + [0]*(maxLen - len(sig1))
    sig2 = sig2 + [0]*(maxLen - len(sig2))
    d = scipy.spatial.distance.euclidean(sig1, sig2)
    if (math.isnan(d)):
       d = 1.0
    return abs(d)
    
def Cosine_dist(sig1, sig2):
    """
    Returns the Cosine distance between graphs described by the signatures
    sig1 and sig2.
    """
    maxLen = max(len(sig1), len(sig2))
    sig1 = sig1 + [0]*(maxLen - len(sig1))
    sig2 = sig2 + [0]*(maxLen - len(sig2))
    d = scipy.spatial.distance.cosine(sig1, sig2)
    if (math.isnan(d)):
       d = 1.0
    #print "cosine = ", d
    return abs(scipy.spatial.distance.cosine(sig1, sig2))



def KulbackLiebler_dist(sig1,sig2): #KulbackLiebler distance   sum(x *log(x/y))
    #maxLen = max(len(sig1), len(sig2))
    #sig1 = sig1 + [0]*(maxLen - len(sig1))
    #sig2 = sig2 + [0]*(maxLen - len(sig2))
    x = np.array(sig1)
    y = np.array(sig2)
    
    d = x*np.log2(x/y)
    d[np.isnan(d)] = 0
    d[np.isinf(d)] = 0
    #print 'KLD ', sum(d)
    return sum(d)

def JensenShannon_dist(sig1,sig2): #JensenShannon distance   sqrt(0.5 * KLD(x, (x+y)/2) + 0.5 * KLD(y, (x+y)/2))
    maxLen = max(len(sig1), len(sig2))
    sig1 = sig1 + [0]*(maxLen - len(sig1))
    sig2 = sig2 + [0]*(maxLen - len(sig2))
    x = np.array(sig1)
    y = np.array(sig2)
    m = (x+y) / 2
    d = math.sqrt(0.5 * KulbackLiebler_dist(x, m) + 0.5 * KulbackLiebler_dist(y, m))
    #print 'JSD', d
    return d
def calcEM(hist1,hist2,h_bins,s_bins):
   #Define number of rows
   numRows = h_bins*s_bins

   sig1 = cv.CreateMat(numRows, 3, cv.CV_32FC1)
   sig2 = cv.CreateMat(numRows, 3, cv.CV_32FC1)    

   for h in range(h_bins):
    for s in range(s_bins): 
        bin_val = cv.QueryHistValue_2D(hist1, h, s)
        cv.Set2D(sig1, h*s_bins+s, 0, cv.Scalar(bin_val))
        cv.Set2D(sig1, h*s_bins+s, 1, cv.Scalar(h))
        cv.Set2D(sig1, h*s_bins+s, 2, cv.Scalar(s))

        bin_val = cv.QueryHistValue_2D(hist2, h, s)
        cv.Set2D(sig2, h*s_bins+s, 0, cv.Scalar(bin_val))
        cv.Set2D(sig2, h*s_bins+s, 1, cv.Scalar(h))
        cv.Set2D(sig2, h*s_bins+s, 2, cv.Scalar(s))

   #This is the important line were the OpenCV EM algorithm is called
   return cv.CalcEMD2(sig1,sig2,cv.CV_DIST_L2)

def EarthMovers_dist(s1,s2):
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
  dis = cv.CalcEMD2(a32,b32,cv.CV_DIST_L1)    #CV_DIST_L2 -- Euclidean Distance, CV_DIST_L1 --- Manhattan Distance 
  return dis


def LineRegress_dist(sig1, sig2):
    #print "In LineRegress Distnace : "
    #print "Sig1 : ", sig1
    #print "Sig2 : ", sig2
    maxLen = max(len(sig1), len(sig2))
    sig1 = sig1 + [0]*(maxLen - len(sig1))
    sig2 = sig2 + [0]*(maxLen - len(sig2))
    slope, intercept, r_value, p_value, std_err = linregress(sig1,sig2)
    r_squared = r_value**2
    #pylab.plot(sig1,sig2)
    #pylab.show()
    #print slope, intercept, r_value, p_value, std_err 

    return 1-r_squared


def n_split(iterable, n, fillvalue=None):
    num_extra = len(iterable) % n
    zipped = zip(*[iter(iterable)] * n)
    return zipped if not num_extra else zipped + [iterable[-num_extra:], ]
    
def JensenShannon_dist_GroupWise(sig1,sig2): #JensenShannon distance   sqrt(0.5 * KLD(x, (x+y)/2) + 0.5 * KLD(y, (x+y)/2))
    maxLen = max(len(sig1), len(sig2))
    sig1 = sig1 + [0]*(maxLen - len(sig1))
    sig2 = sig2 + [0]*(maxLen - len(sig2))
    sig1groups = []
    sig2groups = []
    for group in n_split(sig1 , 5):
        sig1groups.append(join(group))
    for group in n_split(sig2 , 5):
        sig2groups.append(join(group))
    
    #print "sig1groups", sig1groups
    #print "sig2groups", sig2groups
    sumd = 0
    for i in range(len(sig1groups)):
      x = np.array(sig1groups[i])
      y = np.array(sig2groups[i])
      
      m = (x+y) / 2
      d = math.sqrt(0.5 * KulbackLiebler_dist(x, m) + 0.5 * KulbackLiebler_dist(y, m))
      #print "x: ", x , "y: ", y, "d: ", d
      sumd += d
    print 'GroupwiseJSD', sumd
    return sumd

def plotKCore(g) :
    coreness = GraphBase.coreness(g)
    highestcore = max(coreness)
    g.vs["coreness"] = coreness
    g.vs["label"] = g.vs["name"]
    #print("g.vs[name]", g.vs["name"])
    #print("g.vs[coreness]", g.vs["coreness"])
    #print("Highest Core: ", highestcore)
    pal = RainbowPalette()
    pal = {0:"grey", 1:"white", 2:"green",3:"yellow", 4:"red",5:"pink", 6:"black"}
    color_dict = { i: pal.get(i) for i in range(highestcore+1)}
    #print("colordict", color_dict)
    g.vs["color"] = [color_dict[c] for c in g.vs["coreness"]]
    #print("g.vs[color]", g.vs["color"])
    layout = g.layout("kk")
    plot(g,layout=layout) #, vertex_color= [colors[c] for c in g.vs["coreness"]] )
    return
    
 
 

    
