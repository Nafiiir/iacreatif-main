import os
import cv2
import numpy as np

# dossiers
DATASET_FOLDER = "dataset"
TEST_FOLDER = "new_images"

# paramètres
IMAGE_SIZE = (28, 28)
THRESHOLD = 128


# ========================
# PREPROCESSING
# ========================

def preprocess_image(image_path):

    # acquisition
    image = cv2.imread(image_path)

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binarization
    _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)

    # size normalization
    resized = cv2.resize(binary, IMAGE_SIZE)

    # convert to matrix
    matrix = np.array(resized)

    return matrix


# ========================
# BUILD DATABASE
# ========================

def build_reference_database():

    database = {}

    for digit in os.listdir(DATASET_FOLDER):

        digit_path = os.path.join(DATASET_FOLDER, digit)

        if not os.path.isdir(digit_path):
            continue

        database[digit] = []

        for file in os.listdir(digit_path):

            path = os.path.join(digit_path, file)

            matrix = preprocess_image(path)

            database[digit].append(matrix)

    return database


# ========================
# DISTANCE
# ========================

def matrix_distance(m1, m2):

    return np.linalg.norm(m1 - m2)


# ========================
# COMPARISON
# ========================

def compare_image(matrix, database):

    best_digit = None
    best_distance = float("inf")

    for digit in database:

        for ref_matrix in database[digit]:

            distance = matrix_distance(matrix, ref_matrix)

            if distance < best_distance:

                best_distance = distance
                best_digit = digit

    return best_digit


# ========================
# MAIN PROGRAM
# ========================

def main():

    print("Building reference database...")

    database = build_reference_database()

    print("Database ready")

    print("\nComparing new images...\n")

    for file in os.listdir(TEST_FOLDER):

        path = os.path.join(TEST_FOLDER, file)

        matrix = preprocess_image(path)

        predicted_digit = compare_image(matrix, database)

        print(file, "-> recognized as:", predicted_digit)


main()