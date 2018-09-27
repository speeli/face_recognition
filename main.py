import cv2
import numpy as np
import os
import face_record
import train
import test

def face_capture():
	face_record.record()
	
def train_model():
	train.train()
	
def test_model():
	test.test()
	print('test')

if __name__ == "__main__":
    exit == False
    while exit:
    	print('face-reconition:')
    	print('1: face capture')
    	print('2: train model')
    	print('3: test model')
    	print('0: exit')
    	choose = int(input("choose:"))
    	if(choose == 1):
    		face_capture()
    	if(choose == 2):
    		train_model()
    	if(choose == 3):
    		test_model()
    	if(choose == 0):
    		break