import csv
import sys
import collections
from pprint import pprint

def ComputeProbabilities():
    inputfile = sys.argv[1]
    print 'inputfile: ', inputfile
    D = {}
    with open(inputfile, "rb") as infile:
        reader = csv.reader(infile)
        headers = next(reader)[1:]
        for row in reader:
            D[row[0]] = {key: float(value) for key, value in zip(headers, row[1:])}
    #pprint(D)
    
    DCSameClass = {}
    DCOtherClass = {}
    for G1,G1dists in D.items():
      #print G1
      #print G1dists
      SameClass = {}
      OtherClass = {}
      for G2,dist in G1dists.items():
        if not(G1 == G2):
           if (G1[:2] == G2[:2]):
              SameClass[G2] = dist
           else:
              OtherClass[G2] = dist
      DCSameClass[G1]=SameClass
      DCOtherClass[G1]=OtherClass
    #print 'DCSameClass: ', str(DCSameClass)
    #print 'DCOtherClass: ', str(DCOtherClass)
      
    Probs = {}  
    for G,SameClassGraphList in DCSameClass.items() :
         P = 0
         OtherClassGraphList = DCOtherClass[G]
         print 'Graph: ' ,G
         print 'SameClassGraphList: ',SameClassGraphList
         print 'OtherClassGraphList: ', OtherClassGraphList
         numtimes  = 0
         for G1, d1 in SameClassGraphList.items():
             #print G1,d1
             for G2, d2 in OtherClassGraphList.items():
                 numtimes += 1
                 if (d1 < d2) :
                    P = P +1
         print 'numtimes = ', numtimes, ' P = ' , P
         Probs[G] = P*1.0/numtimes
    print str(Probs)         
    average = sum([Probs[key] for key in Probs])/float(len(Probs))
    print 'average: ', average
    
ComputeProbabilities()




