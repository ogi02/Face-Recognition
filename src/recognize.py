# library imports
import time

# custom functions imports
from face_extract_utils import extract_multiple_faces
from face_embedding_utils import get_face_embedding
from candidate_face_check import check_candidate_faces


# recognize faces
def recognize_faces(train_data, detector, model, filename=None):

	start_time = time.time()

	# extract faces of all candidates
	candidate_faces = extract_multiple_faces(detector, filename=filename)

	print("1", end=' ')
	print("--- %s seconds ---" % (time.time() - start_time))

	# check for faces in the image
	if not candidate_faces:
		return ['No faces detected']

	# get face embeddings of all candidates
	candidate_embeddings = [get_face_embedding(face, model) for face in candidate_faces]

	print("2", end=' ')
	print("--- %s seconds ---" % (time.time() - start_time))

	# check faces for all candidates
	names = check_candidate_faces(train_data, candidate_embeddings)

	print("3", end=' ')
	print("--- %s seconds ---" % (time.time() - start_time))

	return names
