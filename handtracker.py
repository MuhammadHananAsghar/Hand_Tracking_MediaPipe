'''
@author: Muhammad Hanan Asghar
'''

# Libraries
import mediapipe as mp
import numpy as np
import cv2


def HandTracker():
	"""
	Function that returns hand tracker object
	"""
	mp_drawings = mp.solutions.drawing_utils
	mp_hands = mp.solutions.hands
	hand_tracker = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
	return hand_tracker


def drawLandMarks(results, image):
	"""
	Function that draws land marks
	"""
	mp_drawings = mp.solutions.drawing_utils
	mp_hands = mp.solutions.hands
	if results.multi_hand_landmarks:
		for num, hand in enumerate(results.multi_hand_landmarks):
			mp_drawings.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
				# Joints Color
				mp_drawings.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
				# Line Color
				mp_drawings.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
			# TO Detect Which Hand is Detecting
			# if get_hand(num, hand, results):
			# 	text, coords = get_hand(num, hand, results)
			# 	cv2.putText(image, text, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
	return image


def get_hand(index, hand, results):
	'''
	Function that returns hands labels
	'''
	mp_drawings = mp.solutions.drawing_utils
	mp_hands = mp.solutions.hands
	output = None
	for idx, classification in enumerate(results.multi_handedness):
		if classification.classification[0].index == index:

			# Process Results
			label = classification.classification[0].label
			score = classification.classification[0].score
			text = f'{label} {round(score, 2)}'
			
			# Getting Co-ordinates
			coords = tuple(np.multiply(
				np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
				[640, 480]
				).astype(int))
			output = text, coords

	return output
