import Levenshtein
from utils.helperFunctions import getConfig, createJSON, CONFIG_PATH
from PIL import Image
import numpy as np
import re
import matplotlib.pyplot as plt

CONFIG = CONFIG_PATH


def wordposFeatures(dataInstance, save=True):
    text = dataInstance["hOCR"]
    pathToInstance = dataInstance["instanceFolderPath"]
    cols, rows = getConfig("gridSize", CONFIG)

    # NOTE: currently works only for one-paged invoices
    imageHeight, imageWidth = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)',
                                        str(text.find_all("div", class_="ocr_page")[0])).groups()[2:4]
    imageHeight, imageWidth = int(imageHeight), int(imageWidth)
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
        # TODO: add name to dict
        createJSON(f"{pathToInstance}\\wordposFeatures.json", {"wordposFeatures": wordPosList})
    return wordPosList


# OBSOLETE
# def wordposFeaturesOLD(invoicePNGPath):
#     image = Image.open(invoicePNGPath)
#
#     # x from left to right; y from top to bottom
#     cols, rows = getConfig("gridSize", CONFIG)
#     imgWidth, imgHeight = image.size
#     gridWidth = imgWidth // cols
#     gridHeight = imgHeight // rows
#
#     gridText = []
#     for row in range(rows):
#         print(f"Processing row {row + 1}")
#         for col in range(cols):
#             leftX = gridWidth * col
#             leftY = gridHeight * row
#             rightX = gridWidth * (col + 1)
#             rightY = gridHeight * (row + 1)
#             croppedImg = image.crop((leftX, leftY, rightX, rightY))
#
#             textCrop = pytesseract.image_to_string(croppedImg)
#             textCrop = (textCrop.replace("\n", " ").split(" "))
#             textCrop = [text for text in textCrop if len(text) > 0]
#             textCrop = " ".join([f"{text}_{row}_{col}" for text in textCrop])
#             gridText.append(textCrop)
#
#     return " ".join(gridText)

# get OCR words per grid and concatenate name with x,y of grid
# store somewhere (?)


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
        # TODO: add name to dict
        createJSON(f"{pathToInstance}\\zonesegFeatures.json", {"zonesegFeatures": gridLabelJoint})

    return gridLabelJoint, sum(gridLabel), len(gridLabel)


def zonesegSimilarity(zoneseg1, zoneseg2):
    return Levenshtein.distance(zoneseg1, zoneseg2)


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


# data = CustomDataset(getConfig("pathToDataFolder", CONFIG))
# print(wordposFeatures(data.__getitem__(0)))
# print(zonesegFeatures(data.__getitem__(0)))
# print(zonesegSimilarity(zonesegFeatures(data.__getitem__(0))[0],zonesegFeatures(data.__getitem__(1))[0]))
# plotGrid(data.__getitem__(0))
