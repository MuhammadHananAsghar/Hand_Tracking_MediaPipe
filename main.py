'''
@author: Muhammad Hanan Asghar
'''

# Libraries
import mediapipe as mp
from handtracker import HandTracker, drawLandMarks, get_hand
import cv2
import numpy as np
'''
@author: Muhammad Hanan Asghar
'''

import uuid
import os

# Tracker Object
hand_tracker = HandTracker()

# Camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
	success, frame = cap.read()
	if not success:
		break
	# Detections
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# Flip Image Horizontal
	image = cv2.flip(image, 1)
	image.flags.writeable = False
	results = hand_tracker.process(image)
	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		
	image = drawLandMarks(results, image)

	cv2.imshow('Hand Tracking', image)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()