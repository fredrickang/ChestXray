import os
import shutil

_dir = ''
data = []
fileList = os.listdir(_dir)
trainDir = 'train'
valDir = 'val'
testDir = 'test'
label = ['No finding', 'disease']

if trainDir not in fileList:
	os.mkdir(_dir + '/' + trainDir)
	os.mkdir(_dir + '/' + trainDir + '/' + label[0])
	os.mkdir(_dir + '/' + trainDir + '/' + label[1])
	
if valDir not in fileList:
	os.mkdir(_dir + valDir)
	os.mkdir(_dir + '/' + valDir + '/' + label[0])
	os.mkdir(_dir + '/' + valDir + '/' + label[1])
	
if testDir not in fileList:
	os.mkdir(_dir + testDir)
	os.mkdir(_dir + '/' + testDir + '/' + label[0])
	os.mkdir(_dir + '/' + testDir + '/' + label[1])
	
	
if label[0] in fileList:
	data = os.listdir(_dir + '/' + label[0])
	for i, d in enumerate(data):
		if i < (len(data)*0.6):
			shutil.move(_dir + '/' + label[0] + '/' + d, _dir + '/' + trainDir + '/' + label[0] + '/' + d)
		elif i < (len(data)*0.8):
			shutil.move(_dir + '/' + label[0] + '/' + d, _dir + '/' + valDir + '/' + label[0] + '/' + d)
		else:
			shutil.move(_dir + '/' + label[0] + '/' + d, _dir + '/' + testDir + '/' + label[0] + '/' + d)
			
if label[1] in fileList:
	data = os.listdir(_dir + '/' + label[1])
	for i, d in enumerate(data):
		if i < (len(data)*0.6):
			shutil.move(_dir + '/' + label[1] + '/' + d, _dir + '/' + trainDir + '/' + label[1] + '/' + d)
		elif i < (len(data)*0.8):
			shutil.move(_dir + '/' + label[1] + '/' + d, _dir + '/' + valDir + '/' + label[1] + '/' + d)
		else:
			shutil.move(_dir + '/' + label[1] + '/' + d, _dir + '/' + testDir + '/' + label[1] + '/' + d)
