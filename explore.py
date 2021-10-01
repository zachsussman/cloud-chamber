import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import MySQLdb as mdb

connection = mdb.connect("localhost", "cloud", "cloud", "test")

def classify(r):
    l, d, i, n = r
    if l > 30 and d < 0.2 and i > 0.8:
        return "red"
    else:
        return "blue"

a,b=[],[]
with connection as cursor:
    cursor.execute("select length, type from tracks;")
    results = cursor.fetchall()
    
    a = np.array([x for (x, y) in results if y=='a'])
    b = np.array([x for (x, y) in results if y=='b'])
    c = np.array([x for (x, y) in results])
    
##    A = np.concatenate((a, b))
##    B = np.array([z for (x, z, y) in results])
##    colors = [{'a':'red', 'b':'blue'}[y] for (x,z, y) in results]

    print np.mean(a), np.std(a), np.percentile(a, 50)
    print np.mean(b), np.std(b), np.percentile(b, 50)
    print np.mean(c), np.std(c)
##    plt.hist(a, color=['red'], bins =range(0,300,25))
##    plt.xlabel("Energy (keV)")

##    colors = [{'a':'red', 'b':'blue'}[z] for (x,y,z) in results]
##    plt.scatter(A, B, c=colors)
##
##    plt.xlabel("Energy (keV)")
##    plt.ylabel("Divergence")
##    
##    plt.show()

    
connection.close()
