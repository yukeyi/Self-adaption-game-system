'''
a = "123.45"
b = float(a)

f = open('test_validata', 'a')
f.write(str([1,2,3]))
f.write('\n')
f.write(str(1))
f.close()

f = open('test_validata', 'r')
strdata = f.readline()
data = strdata[:-2].split(",")
print(data)
'''
import numpy

my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
numpy.savetxt('new.csv', my_matrix, delimiter = ',')

my_matrix_recovery = numpy.loadtxt(open("new.csv","rb"),delimiter=",",skiprows=0)
print(my_matrix_recovery)