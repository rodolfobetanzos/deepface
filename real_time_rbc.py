from deepface import DeepFace
"""
Parameters:
		db_path (string): facial database path. You should store some .jpg files in this folder.

		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble

		detector_backend (string): opencv, ssd, mtcnn, dlib, retinaface

		distance_metric (string): cosine, euclidean, euclidean_l2

		enable_facial_analysis (boolean): Set this to False to just run face recognition

		source: Set this to 0 for access web cam. Otherwise, pass exact video path.

		time_threshold (int): how many second analyzed image will be displayed

		frame_threshold (int): how many frames required to focus on face

"""


DeepFace.stream(db_path = 'D://rbc//Github//face_recognition//data//people_golden//desarrollo', model_name ='Facenet', detector_backend = 'ssd', distance_metric = 'cosine', enable_face_analysis = False, source = 0, time_threshold = 1, frame_threshold = 1)
