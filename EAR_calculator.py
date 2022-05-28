# Define the function for calculating the Eye Aspect Ratio(EAR)
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
	# Vertical eye landmarks
	a = dist.euclidean(eye[1], eye[5])
	b = dist.euclidean(eye[2], eye[4])
	# Horizontal eye landmarks 
	c = dist.euclidean(eye[0], eye[3])

	# The EAR Equation 
	ear = (a + b) / (2.0 * c)
	return ear


def mouth_aspect_ratio(mouth): 
	x = dist.euclidean(mouth[13], mouth[19])
	y = dist.euclidean(mouth[14], mouth[18])
	z = dist.euclidean(mouth[15], mouth[17])

	mar = (x + y + z) / 3.0
	return mar