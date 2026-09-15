import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import seaborn as sns

def extractLbpFeatures(faceImage):
    '''
        - inputs: faceImage: numpy array which represents a grayscale image of a face
        - outputs: hist: numpy array, histogram of LBP features.
        - description: computes LBP features of the input image and returns a normalized histogram of these features.
    '''
    radius = 1
    nPoints = 8 * radius
    lbp = local_binary_pattern(faceImage, nPoints, radius, method="uniform")
    # uniform LBP always has nPoints+2 possible codes (nPoints+1 uniform patterns plus
    # one bin for non-uniform patterns); this must be fixed rather than derived from the
    # observed max, or histograms for different images can come out with different lengths
    nBins = nPoints + 2
    hist, _ = np.histogram(lbp.ravel(), bins=nBins, range=(0, nBins), density=True)
    return hist

def extractOrbFeatures(faceImage, maxKeypoints=500, descriptorSize=32):
    '''
        - inputs: 1) faceImage: a numpy array which represents a grayscale image of a face
                  2) maxKeypoints: an int representing the maximum number of keypoints to use in ORB algorithm
                  3) descriptorSize: an int representing the size of the descriptor to use in ORB algorithm
        - outputs: an array to represent the ORB descriptors or a zero vector if no keypoints are detected
        - description: extracts ORB features from the input image and returns a flattened and padded array of these features
    '''
    orb = cv2.ORB_create(nfeatures=maxKeypoints)
    keypoints, descriptors = orb.detectAndCompute(faceImage, None)
    
    if descriptors is not None:
        # Flatten and pad the descriptor array
        flattenedDescriptors = descriptors.flatten()
        paddedLength = maxKeypoints * descriptorSize
        paddedDescriptors = np.zeros(paddedLength)
        paddedDescriptors[:len(flattenedDescriptors)] = flattenedDescriptors
        return paddedDescriptors
    else:
        # Return a zero vector if no keypoints are detected
        return np.zeros(maxKeypoints * descriptorSize)

def selectLargestFace(faces):
    '''
        - inputs: faces: an array of (x, y, w, h) tuples returned by CascadeClassifier.detectMultiScale
        - outputs: the (x, y, w, h) tuple with the largest area
        - description: detectMultiScale can return multiple candidate faces with no guarantee the first
          one is the most prominent, so this picks the largest by bounding-box area
    '''
    return max(faces, key=lambda face: face[2] * face[3])

def processEmotionImages(baseFolder, emotions, underSample=True):
    '''
        - inputs: 1) baseFolder: a string path to the dataset
                  2) emotions: the array specifying which emotions in the dataset need to be processed
                  3) underSample: a boolean to limit the number of images per emotion
        - outputs: 1) lbpFeaturesList: an array of LBP features for each image
                   2) orbFeaturesList: an array of ORB features for each image
                   3) yLabels:  an array containing the ground-truth emotion associated with each image in the feature set
                   4) emotionImageCount: a dictionary which is the count of images processed for each emotion
        - description: for each emotion in the dataset, this function processes a specified number of images to extract LBP and ORB features, along with the corresponding emotion label
    '''
    # load the Haar Cascade for face detection
    cascadePath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)
    # initialize empty lists to hold the LBP and ORB features along with the yLabel ground truth
    lbpFeaturesList, orbFeaturesList, yLabels = [], [], []
    # count the total number of images per emotion for undersampling purposes
    emotionImageCount = {emotion: 0 for emotion in emotions}
    # to undersample the neutral class which has 593 images, define a maxImages constant
    maxImages = 100

    # for each image in the CK+ dataset, obtain the LBP and ORB features along with the ground truth emotion
    for emotion in emotions:
        emotionFolder = os.path.join(baseFolder, emotion)
        # sort filenames so which images get undersampled is deterministic across machines/runs
        for filename in sorted(os.listdir(emotionFolder)):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # stop reading further images for the current emotion once the maxImage cap is hit and undersampling is on
                if underSample and emotionImageCount[emotion] >= maxImages:
                    break
                emotionImageCount[emotion] += 1
                imagePath = os.path.join(emotionFolder, filename)
                image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue # skip files that aren't readable as images
                # detect faces in the image
                faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) == 0:
                    continue # skip images where no face is detected
                x, y, w, h = selectLargestFace(faces)
                faceImage = image[y:y+h, x:x+w] # extract the face region
                # obtain the LBP and ORB feature sets from the given image
                lbpFeaturesList.append(extractLbpFeatures(faceImage))
                orbFeaturesList.append(extractOrbFeatures(faceImage))
                # obtain the ground-truth emotion for that image
                yLabels.append(emotion)

    return lbpFeaturesList, orbFeaturesList, yLabels, emotionImageCount

def zScoreNormalization(features, K, C):
    '''
        - inputs: 1) features: an array representing the feature set (LBP, ORB, or Combined) being utilized
                  2) K: a number representing the scaling factor for normalization
                  3) C: a constant number to prevent division by zero
        - outputs: normalizedFeatures: an array, Z-score normalized features.
        - description: This function normalizes a feature set using Z-score normalization, which standardizes the features by removing the mean and scaling to unit variance. The scaling factor K and constant C are used to adjust the normalization process
    '''
    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0)
    normalizedFeatures = K * ((features - mu) / (sigma + C))
    return normalizedFeatures

def featureFusion(lbpFeatures, orbFeatures, K, C, singleAxis=True):
    '''
        - inputs: 1) lbpFeatures: numpy array, LBP features
                  2) orbFeatures: numpy array, ORB features
                  3) K: a number representing the scaling factor
                  4) C: a constant number to prevent division by zero
                  5) singleAxis: bool, flag to concatenate features along a single axis
        - outputs: fusedFeatures: an array representing the combined feature set
        - description: This function first normalizes both LBP and ORB features using Z-score normalization and then concatenates them to form a single, unified feature set. The 'singleAxis' parameter controls the axis along which concatenation occurs
    '''
    # normalize each feature set
    lbpNormalized = zScoreNormalization(lbpFeatures, K, C)
    orbNormalized = zScoreNormalization(orbFeatures, K, C)

    # concatenate the normalized features
    if singleAxis:
        fusedFeatures = np.concatenate((lbpNormalized, orbNormalized), axis=1)
    else:
        fusedFeatures = np.concatenate((lbpNormalized, orbNormalized))

    return fusedFeatures

def displayResults(confusionMatrix, classAccuracies, accuracyScore, classificationReport, emotions, title, emotionImageCount, figuresFolder=None):
    '''
        - inputs: 1) confusionMatrix: an array representing the confusion matrix of the predictions
                  2) classAccuracies: an array representing the accuracies of each class.
                  3) accuracyScore: a float representing the overall accuracy of the model
                  4) classificationReport: a string report generated by sklearn's classification_report function for the model
                  5) emotions: an array of emotion labels used
                  6) title: a string title for the results display representing which feature set is being utilizied
                  7) emotionImageCount: a dictionary count of images trained for each emotion
                  8) figuresFolder: optional string path; if given, the confusion matrix plot is also saved there as a .png
        - outputs: prints the accuracies and classification report, and plots the confusion matrix
        - description: this prints the classification report, class accuracies, and overall accuracy. It also visualizes the confusion matrix using a heatmap
    '''
    # plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if figuresFolder:
        os.makedirs(figuresFolder, exist_ok=True)
        plt.savefig(os.path.join(figuresFolder, f'{title.lower()}_confusion_matrix.png'), bbox_inches='tight')

    # plot classification report, class accuracies, and the total accuracy
    print(f"\n{title} Classification Report:")
    print(classificationReport)
    print(f"\n{title} Class Accuracies:")
    for i, emotion in enumerate(emotions):
        print(f"   {emotion} (images trained: {emotionImageCount[emotion]}) - Accuracy: {(classAccuracies[i] * 100):.2f}%")
    print(f"\n{title} Total Accuracy: {(accuracyScore * 100):.2f}%")
    plt.show()