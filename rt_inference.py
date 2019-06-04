import cv2
import numpy as np
import os, sys
from time import strftime, localtime
import random
from openvino.inference_engine import IENetwork, IEPlugin

plugin = IEPlugin("CPU", "/opt/intel/openvino_2019.1.094/deployment_tools/inference_engine/lib/intel64")

model_xml = '/home/ai/work/caffe/examples/cifar10/cifar.xml'
model_bin = '/home/ai/work/caffe/examples/cifar10/cifar.bin'
print('Loading network files:\n\t{}\n\t{}'.format(model_xml, model_bin))

net = IENetwork(model=model_xml, weights=model_bin)

supported_layers = plugin.get_supported_layers(net)
not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
if len(not_supported_layers) != 0:
	print("Following layers are not supported by the plugin for specified device {}:\n {}".formaT(plugin.device, ', '.join(not_supported_layers)))
	print("Please try to specify cpu extensions library path in sample's command line parameters using –l or --cpu_extension command line argument")
	sys.exit(1)

net.batch_size = 1

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

exec_net = plugin.load(network=net)

cap = cv2.VideoCapture(0)
if not cap.isOpened(): sys.exit('camera error')

name = ['rock', 'paper', 'scissors']
	
while True:
	ret, frame = cap.read()
	if not ret: continue 

	rows, cols, channels = frame.shape
	width = cols
	height = rows
	length = min(width, height)
	pt = [60,60]
	if width < height: pt[1] += int((height - length) / 2)
	else: pt[0] += int((width - length) / 2)
	green = (0, 255, 0)  #BGR
	length -= 120
	cv2.rectangle(frame, (pt[0], pt[1]), (pt[0]+length, pt[1]+length), green, 4)

	ch = cv2.waitKey(1) & 0xFF
	if ch == 27: break

	mid_frame = frame[pt[1]:pt[1]+length, pt[0]:pt[0]+length]
	cut_frame = cv2.resize(mid_frame, (32, 32))
	img = cut_frame
	height, width, _ = img.shape
	n, c, h, w = net.inputs[input_blob].shape
	img2 = img
	if height != h or width != w:
		img2 = cv2.resize(img, (w, h))

	img2 = img2.transpose((2, 0, 1))  
	images = np.ndarray(shape=(n, c, h, w))
	images[0] = img2

	res = exec_net.infer(inputs={input_blob: images})
	probs = res[out_blob]

	id = np.argsort(probs)[0][:-2:-1][0]
	prob = probs[0][id[0]]
	inf_res = ''
	if prob >= 0.7: inf_res = name[id[0]]

	if inf_res != '':
		cv2.putText(frame, inf_res, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), lineType=cv2.LINE_AA)
	cv2.imshow('view', frame)