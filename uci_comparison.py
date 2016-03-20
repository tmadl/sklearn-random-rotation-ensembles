import numpy as np

cols = ["a","b","c"]
rows = ["1","2","3"]
d = np.eye(3,3)
cellsize = 15
row_format =("{:>"+str(cellsize)+"}") * (len(rows) + 1)
print row_format.format("", *cols)
print "".join(["="]*cellsize*(len(cols)+1))
for rh, row in zip(rows, d):
    print row_format.format(rh, *row)