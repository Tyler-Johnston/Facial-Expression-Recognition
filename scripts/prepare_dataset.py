import argparse
import csv
import os

import cv2
import numpy as np

# label mapping used by this project's CSV-format CK+ export (emotion,pixels,Usage columns,
# 48x48 grayscale pixels per row) -- folder names are lowercased to match classification.py's
# expected `datasets/CK+/<emotion>/*.jpg` layout
EMOTION_LABELS = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
    7: "contempt",
}

def convertCsvToImageFolders(csvPath, outputFolder, upscaleSize=192):
    '''
        - inputs: 1) csvPath: path to a ckextended.csv-style file (emotion,pixels,Usage columns)
                  2) outputFolder: destination folder, e.g. 'datasets/CK+'
                  3) upscaleSize: pixel size to resize each 48x48 image to before saving, since the
                     Haar cascade face detector misses faces on a meaningful fraction of images at
                     native 48x48 resolution
        - outputs: None, writes one .jpg per row under outputFolder/<emotion>/
        - description: rebuilds the folder-of-images dataset layout this project's pipeline expects
          from a ckextended.csv-style export
    '''
    counts = {name: 0 for name in EMOTION_LABELS.values()}
    with open(csvPath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion = EMOTION_LABELS[int(row['emotion'])]
            emotionFolder = os.path.join(outputFolder, emotion)
            os.makedirs(emotionFolder, exist_ok=True)

            pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
            if upscaleSize and upscaleSize != 48:
                pixels = cv2.resize(pixels, (upscaleSize, upscaleSize), interpolation=cv2.INTER_CUBIC)

            counts[emotion] += 1
            imagePath = os.path.join(emotionFolder, f'{emotion}_{counts[emotion]:04d}.jpg')
            cv2.imwrite(imagePath, pixels)

    return counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rebuild datasets/CK+/<emotion>/*.jpg from a ckextended.csv-style export')
    parser.add_argument('csvPath', type=str, help='Path to the ckextended.csv-style file')
    parser.add_argument('--output', type=str, default='datasets/CK+', help='Output folder (default: datasets/CK+)')
    args = parser.parse_args()

    counts = convertCsvToImageFolders(args.csvPath, args.output)
    print(f"Wrote images to '{args.output}':")
    for emotion, count in counts.items():
        print(f"  {emotion}: {count}")
