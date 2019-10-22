f = open('../train/SPECT.train', 'r')
data = f.read()

for point in data:
    print('Hi' + point)
f.close()

# f = open('../train/SPECT.train', 'w')
# f.write(data.replace(',', " "))
#
#
#
# f = open('../test/SPECT.test', 'r')
# data = f.read()
# f.close()
#
# f = open('../test/SPECT.test', 'w')
# f.write(data.replace(',', " "))