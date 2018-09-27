import cv2
import numpy as np
import os
import time
def record():
	paused = True
	
	#cap = cv2.VideoCapture('C:/Users/pit12/Desktop/github/face_reconition/women_in_red.mp4')
	cap = cv2.VideoCapture(0)
	if (cap.isOpened()==False):
		print('error cam is broken')
	print('select name of user:')
	name = input()	
	if not (os.path.isdir('images/'+name)):
		os.makedirs('images/'+name)
	list = os.listdir('images/'+name)
	counter = 0
	print(len(list))
	counter = len(list)
	print(counter)
	while (cap.isOpened()):
		ret, frame = cap.read()
		if paused==False :
			
			cv2.imwrite( ('images/'+name+"/"+"img"+str(counter)+".jpg"), frame )
			counter = counter+1
			print('anzahl der bilder: ',counter)
		
		
		cv2.imshow('frame',frame)
		#cv2.imshow('frame',frame)
		if (counter > 512):
			break
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if cv2.waitKey(1) & 0xFF == ord('x'):
			paused = not paused
			if (paused==False):
				paused==True
				print('Paused!')
			else:
				paused==False
				print('Unpaused!')
				time.sleep(0.2)


	cap.release()
	cv2.destroyAllWindows()