import argparse
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import extractLbpFeatures, extractOrbFeatures, featureFusion, selectLargestFace

def loadModel(modelPath):
    '''
        - inputs: modelPath: a string path to the saved model file
        - outputs: 1) model: loaded model object
                   2) featureType: string, the feature set ('LBP', 'ORB', or 'Combined') the model was trained on
        - description: this loads a saved facial classification file, along with the feature type it expects
    '''
    with open(modelPath, 'rb') as file:
        saved = pickle.load(file)
    return saved['model'], saved['featureType']

def preprocessImage(imagePath, faceCascade, featureType):
    '''
        - inputs: 1) imagePath: a string path to the image file.
                  2) faceCascade: OpenCV CascadeClassifier object, used for face detection
                  3) featureType: a string which represents the type of features to extract ('LBP', 'ORB', or 'Combined')
        - outputs: features: an array representing the extracted features from the face image
        - description: This function reads an image from 'imagePath', detects faces using 'faceCascade', and extracts features (LBP, ORB, or combined) from the detected face based on 'featureType'
    '''
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image file: '{imagePath}'")

    # detect faces in the image
    faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # extract the largest detected face from the provided image file
    x, y, w, h = selectLargestFace(faces)
    faceImage = image[y:y+h, x:x+w]

    # extract LBP and ORB features from the face image
    lbpFeatures = extractLbpFeatures(faceImage)
    orbFeatures = extractOrbFeatures(faceImage)

    # determine which features to use
    if featureType == 'Combined':
        combinedFeatures = featureFusion(lbpFeatures, orbFeatures, K=100, C=1, singleAxis=False)
        return combinedFeatures
    elif featureType == 'ORB':
        return orbFeatures
    else:
        return lbpFeatures

def main(imagePath, modelPath):
    '''
        - inputs: 1) imagePath: str, path to the image file
                  2) modelPath: str, path to the saved model file
        - outputs: None, but displays the image with the predicted emotion and decision scores
        - description: loads a pre-trained model and an image, processes the image to extract features, predicts the emotion using the model, and displays the image with the predicted emotion and top decision scores
    '''
    # load the Haar Cascade for face detection
    cascadePath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)

    # load model and preprocess the image
    model, featureType = loadModel(modelPath)
    features = preprocessImage(imagePath, faceCascade, featureType)

    # reshape features for single sample prediction
    features = features.reshape(1, -1)

    # make a prediction and get decision function scores
    prediction = model.predict(features)
    decisionScores = model.decision_function(features)

    # get the indices of the top predictions based on decision scores
    topIndices = np.argsort(decisionScores[0])[-4:]

    # get the decision scores for the top predictions
    topScores = decisionScores[0][topIndices]

    # get the emotion labels for the top predictions
    topEmotions = model.classes_[topIndices]

    # display the image and the top predictions with their decision scores
    image = cv2.imread(imagePath)
    if image is None:
        raise FileNotFoundError(f"Could not read image file: '{imagePath}'")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(f"Predicted Emotion: {prediction[0]}")
    for i, (emotion, score) in enumerate(zip(topEmotions, topScores)):
        plt.text(10, 30 + 30*i, f"{emotion}: {score:.2f}", fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
    plt.axis('off')
    plt.show()

# USAGE: python3 main.py path_to_image path_to_model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Emotion Recognition')
    parser.add_argument('imagePath', type=str, help='Path to the image file')
    parser.add_argument('modelPath', type=str, help='Path to the model file')
    args = parser.parse_args()
    try:
        main(args.imagePath, args.modelPath)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
