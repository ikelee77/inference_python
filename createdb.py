import lmdb
import sys
import os
import random
import cv2

caffe_home = '/home/intel/caffe/'
sys.path.insert(0, caffe_home + 'python')
import caffe

data_path = '/home/intel/caffe/data/cifar10'
train_db = '/home/intel/caffe/examples/cifar10/cifar10_train_lmdb'
test_db = '/home/intel/caffe/examples/cifar10/cifar10_test_lmdb'

name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'rock', 'paper', 'scissors']

rmt = 'exec rm –r '
rmpath = rmt+train_db
try:
	os.rmdir(rmpath)
except OSError as ex: print(end='')
rmpath = rmt+test_db
try:
	os.rmdir(rmpath)
except OSError as ex: print(end='')

filelist = dict()
for nm in name: filelist[nm] = list()

count = 0
for nm in name:
	work_dir = data_path + '/' + nm
	if not os.path.exists(work_dir): print('path does not exist')
	filenames = os.listdir(work_dir)
	for filename in filenames:
		full_filename = os.path.join(work_dir, filename)
		count += 1
		if not os.path.isdir(full_filename):
			filelist[nm].append(filename)
print("count: {}".format(count))

rate = int(input('testing rate (1~100) : '))

lmdb1  = lmdb.open(train_db, map_size=999999999)
wr1    = lmdb1.begin(write=True)
lmdb2  = lmdb.open(test_db, map_size=999999999)
wr2    = lmdb2.begin(write=True)

datum = caffe.proto.caffe_pb2.Datum()
train_id = 0
test_id = 0
len_id = 0
random.seed(a=None)
height = 32
width = 32

for nm in name:
	test_count = int(rate / 100.0 * len(filelist[nm]))
	random.shuffle(filelist[nm])
	label = nm
	test_pos = 0;
	for fc in filelist[nm]:
		fn = data_path + '/' + nm + '/' + fc                   
		img = cv2.imread(fn, 1) # read bytes
		img = cv2.resize(img, (height, width)) # resize to 32*32
		_, img_jpg = cv2.imencode('.jpg', img) # encode to png
		# img_jpg = img
		datum.channels = 3
		datum.height = height
		datum.width = width
		datum.label = name.index(nm)
		datum.encoded = True
		datum.data = img_jpg.tostring()
		datum_byte = datum.SerializeToString()

		if test_pos < test_count:
			test_pos += 1
			index_byte = '%010d' % test_id
			test_id += 1
			wr2.put(index_byte.encode('ascii'), datum_byte, append=True)  # byte img data + ascii index
		else:
			index_byte = '%010d' % train_id
			train_id += 1
			wr1.put(index_byte.encode('ascii'), datum_byte, append=True)


wr1.commit()
wr2.commit()
lmdb1.close()
lmdb2.close()
