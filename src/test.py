# library imports
import os
import sys

from numpy import load
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace

# project imports
from recognize import recognize_faces

MODEL_NAME = "vgg16"
EMBEDDING_FILE = "../models/embeddings.npz"
TEST_FOLDER = "../dataset/test/"


if __name__ == "__main__":
    # load known face embeddings
    data = load(EMBEDDING_FILE, allow_pickle=True)
    train_data = data["trainData"].item()

    # initialize mtcnn detector
    detector = MTCNN()

    # initialize vggface model
    model = VGGFace(model=MODEL_NAME, include_top=False, input_shape=(160, 160, 3), pooling="avg")

    result = []
    # check if the folder exists
    if os.path.exists(TEST_FOLDER) and os.path.isdir(TEST_FOLDER):
        # iterate through all files and subdirectories in the folder
        for folder_name in os.listdir(TEST_FOLDER):
            # construct path to each person
            path_to_person = os.path.join(TEST_FOLDER, folder_name)

            if os.path.isdir(path_to_person):
                # iterate through all faces for a person
                for file_name in os.listdir(path_to_person):
                    # construct file name
                    full_path = os.path.join(path_to_person, file_name)

                    if os.path.isfile(full_path):
                        # recognize face
                        names = recognize_faces(train_data, detector, model, full_path)
                        result.append({
                            "file_path": full_path,
                            "predicted": names[0].lower().replace(" ", "_"),
                            "actual": full_path.split("/")[-1].split("\\")[0]
                        })
    else:
        print("The specified folder does not exist or is not a directory.")

    # print result
    print("\n\n----- Results -----")
    for i, r in enumerate(result):
        print(f"{i}. {r.get('file_path').split('/')[-1]} -> {r.get('predicted')}")

    # test count
    test_count = len(result)
    # predicted count
    predicted_count = len([r for r in result if r.get("predicted") == r.get("actual")])
    # percentage
    percentage = ("{:.2f}".format(predicted_count / test_count * 100))

    # print stats
    print("\n----- Stats -----")
    print(f"Test count - {test_count}")
    print(f"Predicted count - {predicted_count}")
    print(f"Percentage - {percentage}")

    # redirect stdout
    orig_stdout = sys.stdout
    file = open("result-test-with-trained-data.txt", "w")
    sys.stdout = file

    # save to file
    print("----- Results -----")
    for i, r in enumerate(result):
        print(f"{i+1}. {r.get('file_path').split('/')[-1]} -> {r.get('predicted')}")
    print("\n----- Stats -----")
    print(f"Test count - {test_count}")
    print(f"Predicted count - {predicted_count}")
    print(f"Percentage - {percentage}")

    # return stdout
    sys.stdout = orig_stdout
