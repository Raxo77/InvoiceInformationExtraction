import os
import re
import math
import string
import numpy as np
import pandas as pd
from PIL import Image
from decimal import Decimal
from utils.helperFunctions import getConfig, CONFIG_PATH


def textualFeatures(wordsInfo, vicinityThreshold=4):
    # topmost and bottommost words - the same for every word in the invoice
    topmost = f"{wordsInfo[0][0]}_{wordsInfo[0][1][0]}_{wordsInfo[0][1][1]}"
    bottommost = f"{wordsInfo[-1][0]}_{wordsInfo[-1][1][0]}_{wordsInfo[-1][1][1]}"

    # create dict to store textual features
    wordContext = {
        f"{i[0]}_{i[1][0]}_{i[1][1]}": {"topmost": topmost, "bottommost": bottommost, "left": "", "right": "",
                                        "above": "", "below": ""} for i in wordsInfo}

    for wordCount, temp in enumerate(wordsInfo):
        word, coords = temp

        maxDistance = (coords[3] - coords[1]) * vicinityThreshold

        # left word
        if wordCount != 0:
            if wordsInfo[wordCount - 1][1][1] == coords[1] or wordsInfo[wordCount - 1][1][3] == coords[3]:
                wordContext[f"{word}_{coords[0]}_{coords[1]}"][
                    "left"] = f"{wordsInfo[wordCount - 1][0]}_{wordsInfo[wordCount - 1][1][0]}_{wordsInfo[wordCount - 1][1][1]}"

        # right word
        if wordCount != len(wordsInfo) - 1:
            if wordsInfo[wordCount + 1][1][1] == coords[1] or wordsInfo[wordCount + 1][1][3] == coords[3]:
                wordContext[f"{word}_{coords[0]}_{coords[1]}"][
                    "right"] = f"{wordsInfo[wordCount + 1][0]}_{wordsInfo[wordCount + 1][1][0]}_{wordsInfo[wordCount + 1][1][1]}"

        # above and below words
        # Notably, there may be cases where not above or below word was found
        # -- one could consider increasing the size of the coordinate window
        # to always find a "nearest" above or below word but this would
        # defeat the purpose of capturing local context

        aboveWord = ""
        belowWord = ""
        midPointFocalWord = ((coords[0] + coords[2]) / 2,
                             (coords[1] + coords[3]) / 2)
        minDistanceAbove = math.inf
        minDistanceBelow = math.inf
        for otherWord, otherCoords in wordsInfo:

            if coords[1] > otherCoords[3]:
                midPointCandidateWord = ((otherCoords[0] + otherCoords[2]) / 2,
                                         (otherCoords[1] + otherCoords[3]) / 2)
                euclDistance = math.sqrt((midPointFocalWord[0] - midPointCandidateWord[0]) ** 2 + (
                        midPointFocalWord[1] - midPointCandidateWord[1]) ** 2)
                if euclDistance < minDistanceAbove and euclDistance < maxDistance:
                    minDistanceAbove = euclDistance
                    aboveWord = f"{otherWord}_{otherCoords[0]}_{otherCoords[1]}"

            elif coords[3] < otherCoords[1]:
                midPointCandidateWord = ((otherCoords[0] + otherCoords[2]) / 2,
                                         (otherCoords[1] + otherCoords[3]) / 2)
                euclDistance = math.sqrt((midPointFocalWord[0] - midPointCandidateWord[0]) ** 2 + (
                        midPointFocalWord[1] - midPointCandidateWord[1]) ** 2)
                if euclDistance < minDistanceBelow and euclDistance < maxDistance:
                    minDistanceBelow = euclDistance
                    belowWord = f"{otherWord}_{otherCoords[0]}_{otherCoords[1]}"

        wordContext[f"{word}_{coords[0]}_{coords[1]}"]["above"] = aboveWord
        wordContext[f"{word}_{coords[0]}_{coords[1]}"]["below"] = belowWord

    return wordContext


def layoutFeatures(wordsInfo, imageSize):
    layoutFeatures = {
        f"{i[0]}_{i[1][0]}_{i[1][1]}": {"normTop": -1, "normLeft": -1, "normBottom": -1, "normRight": -1,
                                        "wordWidth": -1, "wordHeight": -1, "wordArea": -1} for i in
        wordsInfo}

    imageWidth, imageHeight = imageSize

    # calculate normalised position of word on the page and ngram width, height and area
    for word, coords in wordsInfo:
        layoutFeatures[f"{word}_{coords[0]}_{coords[1]}"]["normTop"] = round(coords[1] / imageHeight, 4)
        layoutFeatures[f"{word}_{coords[0]}_{coords[1]}"]["normLeft"] = round(coords[0] / imageWidth, 4)
        layoutFeatures[f"{word}_{coords[0]}_{coords[1]}"]["normBottom"] = round(coords[3] / imageHeight, 4)
        layoutFeatures[f"{word}_{coords[0]}_{coords[1]}"]["normRight"] = round(coords[2] / imageHeight, 4)
        layoutFeatures[f"{word}_{coords[0]}_{coords[1]}"]["wordWidth"] = coords[2] - coords[0]
        layoutFeatures[f"{word}_{coords[0]}_{coords[1]}"]["wordHeight"] = coords[3] - coords[1]
        layoutFeatures[f"{word}_{coords[0]}_{coords[1]}"]["wordArea"] = (coords[2] - coords[0]) * (
                coords[3] - coords[1])

    return layoutFeatures


def patternFeatures(wordsInfo):
    patternFeatures = {
        f"{i[0]}_{i[1][0]}_{i[1][1]}": {"standardisedText": ""} for i in wordsInfo}

    for word, coords in wordsInfo:

        # email
        emailPattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        urlPattern = r"^https?://[\w\.-]+(\.[\w\.-]+)+[/\w\.-]*$"
        if re.match(emailPattern, word):
            patternFeatures[f"{word}_{coords[0]}_{coords[1]}"]["standardisedText"] = "<EMAIL>"

        # URL
        elif re.match(urlPattern, word):
            patternFeatures[f"{word}_{coords[0]}_{coords[1]}"]["standardisedText"] = "<URL>"

        # conventional string
        else:
            standardisedText = []
            for char in list(word):
                # digit
                if char.isnumeric():
                    standardisedText.append("d")
                # char upper-case or lower-case
                elif char.isalpha():
                    standardisedText.append("c") if char.islower() else standardisedText.append("C")
                # punctuation et sim.
                else:
                    standardisedText.append("p")

            patternFeatures[f"{word}_{coords[0]}_{coords[1]}"]["standardisedText"] = "".join(standardisedText)

    return patternFeatures


def logicFeatures(wordsInfo, hOCR, textualFeatures, numNGrams=4, titleThreshold=0.38):
    # Notably, the latter three features are mutually exclusive

    if len([titleThreshold]) < 2:
        titleThreshold = [titleThreshold, titleThreshold]
    logicFeatures = {
        f"{i[0]}_{i[1][0]}_{i[1][1]}": {"isTitle_heightBased": False, "isTitle_hOCRBased": False, "isSum": False,
                                        "isProduct": False, "isMathElement": False} for i
        in wordsInfo}

    """
    as the hOCR output contains no direct information on the font size, the height
    of each token's bounding box will be used as proxy. To then assess, whether a toke
    is to be considered a title, the height of the bonding box will be compared to the average
    height of all bounding boxes in the document (maybe including a certain tolerance level)
    """

    # 2 approaches and check which on better: 1st use avg bb size; 2nd use avg x_size
    fontSizeProxy_heightBased = sum([coords[3] - coords[1] for _, coords in wordsInfo]) / len(wordsInfo)

    fontSizes = []
    for line in hOCR.find_all(class_="ocr_line"):
        x_sizeList = [element for element in line.get("title").split(";") if "x_size" in element]
        fontSizes.append(float(x_sizeList[0].split('x_size')[1].strip()))

    fontSizeProxy_hOCRBased = sum(fontSizes) / len(fontSizes)

    for word, coords in wordsInfo:

        # isTitle
        if coords[3] - coords[1] > (1 + titleThreshold[0]) * fontSizeProxy_heightBased:
            logicFeatures[f"{word}_{coords[0]}_{coords[1]}"]["isTitle_heightBased"] = True
        if coords[3] - coords[1] > (1 + titleThreshold[1]) * fontSizeProxy_hOCRBased:
            logicFeatures[f"{word}_{coords[0]}_{coords[1]}"]["isTitle_hOCRBased"] = True

        # isPartOfMathOperation
        # Analyse n-grams (the paper uses trigrams), horizontally or vertically aligned

        if word.replace(".", "").replace(",", "").isnumeric():
            nGramSequence = {"above": [f"{word}_{coords[0]}_{coords[1]}"], "right": [f"{word}_{coords[0]}_{coords[1]}"]}
            for direction in ("right", "above"):
                nGramCounter = numNGrams
                while nGramCounter > 0:
                    candidate = textualFeatures[nGramSequence[direction][-1]][direction]
                    if candidate != "" and candidate.split("_")[0].replace(".", "").replace(",", "").isnumeric():
                        nGramSequence[direction].append(candidate)

                    nGramCounter -= 1
                if len(nGramSequence[direction]) > 2:
                    # ASSUMING ONLY A LEFT TO RIGHT AND BOTTOM-DOWN CALCULATION METHOD:
                    if direction == "above":
                        result = Decimal(nGramSequence[direction][0].split("_")[0].replace(".", "").replace(",", ""))
                        summation = sum([Decimal(i.split("_")[0].replace(".", "").replace(",", "")) for i in
                                         nGramSequence[direction][1:]])
                        product = np.product([Decimal(i.split("_")[0].replace(".", "").replace(",", "")) for i in
                                              nGramSequence[direction][1:]])

                        if result == summation:
                            logicFeatures[nGramSequence[direction][0]]["isSum"] = True
                            for element in nGramSequence[direction][1:]:
                                logicFeatures[element]["isMathElement"] = True

                        if result == product:
                            logicFeatures[nGramSequence[direction][0]]["isProduct"] = True
                            for element in nGramSequence[direction][1:]:
                                logicFeatures[element]["isMathElement"] = True


                    # direction == "above"
                    else:
                        result = Decimal(nGramSequence[direction][-1].split("_")[0].replace(".", "").replace(",", ""))
                        summation = sum([Decimal(i.split("_")[0].replace(".", "").replace(",", "")) for i in
                                         nGramSequence[direction][:-1]])
                        product = np.product([Decimal(i.split("_")[0].replace(".", "").replace(",", "")) for i in
                                              nGramSequence[direction][:-1]])

                        if result == summation:
                            logicFeatures[nGramSequence[direction][-1]]["isSum"] = True
                            for element in nGramSequence[direction][:-1]:
                                logicFeatures[element]["isMathElement"] = True

                        if result == product:
                            logicFeatures[nGramSequence[direction][-1]]["isProduct"] = True
                            for element in nGramSequence[direction][:-1]:
                                logicFeatures[element]["isMathElement"] = True

        # up to this point only those list "survive" where each ngram is numeric
        # now, firstly filter those that are only of length 1; then for each ngram of up to length numNGrams:
        # check whether it is the result or an element in a mathematical operation

    return logicFeatures


def deriveFeatures(dataInstance, includePunct, save=True, vicinityThreshold=4):
    hOCR = dataInstance["hOCR"]

    # wordsInfo serves as basis for the derivation of additional features. It contains the words identified by the
    # OCR engine and the respective coordinates of each word's bounding box
    wordsInfo = []
    for word in hOCR.find_all("span", class_="ocrx_word"):
        lexem = word.get_text().replace("\n", "").replace(" ", "")
        if not includePunct:

            for char in string.punctuation:
                lexem = lexem.replace(char, "")
            if len(lexem) == 0:
                continue
        coords = word["title"].split(";")[0].split(" ")[1:]
        coords = [int(i) for i in coords]
        wordsInfo.append((lexem, coords))

    # topmost, bottommost, left, right, above and below word
    textualFeaturesDict = textualFeatures(wordsInfo, vicinityThreshold)

    # get tuple of image width, height
    imageSize = Image.open(dataInstance["pathInvoicePNG"]).size

    # layout features
    layoutFeatureDict = layoutFeatures(wordsInfo, imageSize)

    # patternFeatures
    patternFeatureDict = patternFeatures(wordsInfo)

    logicFeaturesDict = logicFeatures(wordsInfo, hOCR, textualFeaturesDict, )
    print()

    jointDict = {
        key: values for key, values in textualFeaturesDict.items()}
    for key in jointDict.keys():
        for temp in {1: layoutFeatureDict, 2: patternFeatureDict, 3: logicFeaturesDict}.values():
            for k, val in temp[key].items():
                jointDict[key][k] = val

    # , layoutFeatureDict[key], patternFeatureDict[key], logicFeaturesDict[key]] for
    # key in textualFeaturesDict.keys()
    df = pd.DataFrame.from_dict(data=jointDict, orient="index")
    df = df.reset_index().rename(columns={"index": "wordKey"})

    if save and includePunct:
        df.to_csv(os.path.join(dataInstance["instanceFolderPath"], "BERT_features.csv"), index=False)
    elif save and not includePunct:
        df.to_csv(os.path.join(dataInstance["instanceFolderPath"], "BERT_features_noPunct.csv"), index=False)

    return df