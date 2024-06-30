import re
import math
import numpy as np
import Levenshtein
from PIL import Image
import matplotlib.pyplot as plt
from utils.helperFunctions import getConfig, createJSON, CONFIG_PATH

CONFIG = CONFIG_PATH


def wordposFeatures(dataInstance, save=True):
    text = dataInstance["hOCR"]
    pathToInstance = dataInstance["instanceFolderPath"]
    cols, rows = getConfig("gridSize", CONFIG)

    imageHeight, imageWidth = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)',
                                        str(text.find_all("div", class_="ocr_page")[0])).groups()[2:4]
    imageWidth, imageHeight = int(imageHeight), int(imageWidth)

    gridHeight = imageHeight // rows
    gridWidth = imageWidth // cols

    wordPosList = []
    for word in text.find_all("span", class_="ocrx_word"):
        bbox_X, bbox_Y = word["title"].split(";")[0].split(" ")[1:3]
        bbox_X, bbox_Y = int(bbox_X), int(bbox_Y)
        colNum = bbox_X // gridWidth
        rowNum = bbox_Y // gridHeight
        wordText = word.get_text().replace("\n", "").replace(" ", "")
        wordPos = f"{wordText}_{rowNum}_{colNum}"
        wordPosList.append(wordPos)

    if save:
        createJSON(f"{pathToInstance}\\wordposFeatures.json", {"wordposFeatures": wordPosList})
    return wordPosList


def zonesegFeatures(dataInstance, threshold=.1, save=True):
    image = dataInstance["pathInvoicePNG"]
    image = Image.open(image).convert("L")
    imageArray = np.array(image)

    imageArray = np.where(imageArray > 127, 255, 0)
    pathToInstance = dataInstance["instanceFolderPath"]

    cols, rows = getConfig("gridSize", CONFIG)
    gridWidth = imageArray.shape[1] // cols
    gridHeight = imageArray.shape[0] // rows

    gridLabel = []
    for row in range(rows):
        for col in range(cols):
            xStart = row * gridHeight
            xEnd = (row + 1) * gridHeight
            yStart = col * gridWidth
            yEnd = (col + 1) * gridWidth

            subArray = imageArray[xStart:xEnd, yStart:yEnd]

            whitePixels = np.sum(subArray == 0)
            allPixels = subArray.shape[0] * subArray.shape[1]

            if whitePixels / allPixels >= threshold:
                gridLabel.append(1)
            else:
                gridLabel.append(0)

    gridLabelJoint = "".join([str(i) for i in gridLabel])
    if save:
        createJSON(f"{pathToInstance}\\zonesegFeatures.json", {"zonesegFeatures": gridLabelJoint})

    return gridLabelJoint, sum(gridLabel), len(gridLabel)


def zonesegSimilarity(zoneseg1, zoneseg2):
    count = 0
    for i, j in zip(list(zoneseg1), list(zoneseg2)):
        if i != j:
            count += 1
        else:
            count += 0
    return -math.sqrt(count)


def wordposSimilarity(wordpos1, wordpos2):
    return -Levenshtein.distance(wordpos1, wordpos2)


def plotGrid(dataInstance):
    image = Image.open(dataInstance["pathInvoicePNG"])

    imageWidth, imageHeight = image.size
    fig, grid = plt.subplots()
    grid.imshow(image)
    cols, rows = getConfig("gridSize", CONFIG)

    gridHeight = imageHeight // rows
    gridWidth = imageWidth // cols

    grid.set_xticks(range(0, imageWidth, gridWidth))
    grid.set_yticks(range(0, imageHeight, gridHeight))
    grid.grid()

    plt.show()

