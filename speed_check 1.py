import cv2
import dlib
import time
import threading
import math
import argparse
from scipy.spatial import distance as dist
import numpy as np
# construct the argument parser and parse the arguments


carCascade = cv2.CascadeClassifier('cars.xml')
video = cv2.VideoCapture("cars.mp4")

WIDTH = 1280
HEIGHT = 720


def estimateSpeed(loc1_0, loc1_1,loc2_0,loc2_1,t):
	d=dist.euclidean((loc1_0,loc1_1),(loc2_0,loc2_1))
	#d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	# ppm = location2[2] / carWidht
	#ppm = 8.8
	speed = -d / (t*100)
	#print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
	#fps = 18
	return speed
def chang_n_d(frame):
	arr=frame*np.array([0.1,0.2,0.5])
	arr2=(255*arr/arr.max())
	return arr2

def trackMultipleObjects():
	rectangleColor = (0, 255, 0)
	frameCounter = 0
	currentCarID = 0
	fps = 0
	
	carTracker = {}
	carNumbers = {}
	carLocation1 = {}
	carLocation2 = {}
	speed = 0
	told=-1
	tnew=-1
	tnewus=-1
	tdef=-1
	x=False
	
	# Write output to video file
	#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH,HEIGHT))


	while True:
		
		#start_time = time.time()
		if x:
			told=tnew
		tnew=cv2.getTickCount()
		rc, image = video.read()
		if type(image) == type(None) :
			tnewus=tnew
			break
		
		image = cv2.resize(image, (WIDTH, HEIGHT))
		resultImage = image.copy()
		
		frameCounter = frameCounter + 1
		
		carIDtoDelete = []

		for carID in carTracker.keys():
			trackingQuality = carTracker[carID].update(image)
			
			if trackingQuality < 7:
				carIDtoDelete.append(carID)
				
		for carID in carIDtoDelete:
			print ('Removing carID ' + str(carID) + ' from list of trackers.')
			print ('Removing carID ' + str(carID) + ' previous location.')
			print ('Removing carID ' + str(carID) + ' current location.')
			carTracker.pop(carID, None)
			carLocation1.pop(carID, None)
			carLocation2.pop(carID, None)
		
		if not (frameCounter % 10):
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
			
			for (_x, _y, _w, _h) in cars:
				x = int(_x)
				y = int(_y)
				w = int(_w)
				h = int(_h)
			
				x_bar = x + 0.5 * w
				y_bar = y + 0.5 * h
				
				matchCarID = None
			
				for carID in carTracker.keys():
					trackedPosition = carTracker[carID].get_position()
					
					t_x = int(trackedPosition.left())
					t_y = int(trackedPosition.top())
					t_w = int(trackedPosition.width())
					t_h = int(trackedPosition.height())
					
					t_x_bar = t_x + 0.5 * t_w
					t_y_bar = t_y + 0.5 * t_h
				
					if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
						matchCarID = carID
					
				
				if matchCarID is None:
					print ('Creating new tracker ' + str(currentCarID))
					
					tracker = dlib.correlation_tracker()
					tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
					
					carTracker[currentCarID] = tracker
					carLocation1[currentCarID] = [x, y, w, h]

					currentCarID = currentCarID + 1
		
		cv2.line(resultImage,(0,480),(1280,480),(0,255,0),5)
		if len(carTracker)>0:
			x=False
		else:
			x=True
			tnewus=tnew
			
		if tnewus>0 & tnew>0:
			tdef=tnewus-tnew


		for carID in carTracker.keys():
			trackedPosition = carTracker[carID].get_position()
					
			t_x = int(trackedPosition.left())
			t_y = int(trackedPosition.top())
			t_w = int(trackedPosition.width())
			t_h = int(trackedPosition.height())
			
			cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
			
			# speed estimation
			carLocation2[carID] = [t_x, t_y, t_w, t_h]

		#end_time = time.time()
		
		#if not (end_time == start_time):
			#fps = 1.0/(end_time - start_time)
		
		#cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (0,50),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        
		for i in carLocation1.keys():	
			
			[x1, y1, w1, h1] = carLocation1[i]
			[x2, y2, w2, h2] = carLocation2[i]
		
				# print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
			carLocation1[i] = [x2, y2, w2, h2]
		
				# print 'new previous location: ' + str(carLocation1[i])
			if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
					
				speed = estimateSpeed(x1, y1, x2, y2 ,tdef)
				cv2.putText(resultImage, str(int(speed)) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)				
					#print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')

			else:
				cv2.putText(resultImage, "Far Object", (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

						#print ('CarID ' + str(i) + ' Location1: ' + str(carLocation1[i]) + ' Location2: ' + str(carLocation2[i]) + ' speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
		cv2.imshow('result', resultImage)
        
		# Write the frame into the file 'output.avi'
		#out.write(resultImage)


		if cv2.waitKey(33) == 27:
			break
	
	cv2.destroyAllWindows()

if __name__ == '__main__':
	trackMultipleObjects()
