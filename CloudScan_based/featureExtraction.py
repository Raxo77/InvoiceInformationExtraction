import re
import math
import numpy as np
import pandas as pd
from PIL import Image
from dateutil.parser import parse, ParserError
from dataProcessing.customDataset import CustomDataset
from utils.helperFunctions import getConfig, CONFIG_PATH


def featureCalculation(nGramList: list, dataInstance, citiesGazetteer=pd.DataFrame([]),
                       countryGazetteer=pd.DataFrame([]), ZIPCodesGazetteer=pd.DataFrame([])):

    # (focalWord | stringFeatures | numericFeatures | booleanFeatures | wholeNGram)
    # for each element in the final list
    nGramFeatures = []

    featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
    colNames = list(featuresDF.columns)
    colNames[0] = "wordKey"
    featuresDF.columns = colNames

    # get tuple of image width, height
    imgWidth, imgHeight = Image.open(dataInstance["pathInvoicePNG"]).size

    # citiesGazetteer = pd.read_csv(getConfig("pathToCityGazetteer", CONFIG_PATH), sep="\t")
    # countryGazetteer = pd.read_csv(getConfig("pathToCountryGazetteer", CONFIG_PATH))
    # ZIPCodesGazetteer = pd.read_csv(getConfig("pathToZIPGazetteer", CONFIG_PATH), header=None, sep="\t")

    # calculate the features as described in CloudScanPaper
    # nGramList is a list of sets containing ngram, focal word pairs
    for wordSet in nGramList:
        nGram, word = wordSet
        focalFeatures = (word, [], [], [], nGram)

        # TEXTUAL FEATURES

        rawText = word.split("_")[0]
        focalFeatures[1].append(rawText)

        rawTextLastWord = nGram[-1].split("_")[0]
        focalFeatures[1].append(rawTextLastWord)

        # take the raw text of the word two words left of the (first word in the) ngram - would be alterable
        rawTextTwoWordsLeft = None
        nextWordLeft = featuresDF.loc[featuresDF["wordKey"] == word, "left"].item()
        contextCount = 1
        while not pd.isna(nextWordLeft) and contextCount < 2:
            nextWordLeft = featuresDF.loc[featuresDF["wordKey"] == nextWordLeft, "left"].item()
            contextCount += 1
        if contextCount == 2 and nextWordLeft is not np.nan:
            rawTextTwoWordsLeft = nextWordLeft
        focalFeatures[1].append(rawTextTwoWordsLeft)

        # get the standardised text of the focal word/first word in the ngram; notably, extending this to the entire
        # ngram may be an aspect interesting to investigate in light of performance
        textPatterns = featuresDF.loc[featuresDF["wordKey"] == word, "standardisedText"].item()
        focalFeatures[1].append(textPatterns)

        # NUMERIC FEATURES

        # when splitting *word* by "_" - [0] is word; [1] is x coords, [2] is y coords (upper left corner of bbox)

        bottomMargin = (imgHeight - int(word.split("_")[-1]) + int(
            featuresDF.loc[featuresDF["wordKey"] == word, "wordHeight"].item())) / imgHeight
        focalFeatures[2].append(bottomMargin)

        topMargin = int(word.split("_")[-1]) / imgHeight
        focalFeatures[2].append(topMargin)

        rightMargin = (imgWidth - int(word.split("_")[-2]) - int(
            featuresDF.loc[featuresDF["wordKey"] == word, "wordWidth"].item())) / imgWidth
        focalFeatures[2].append(rightMargin)

        leftMargin = int(word.split("_")[-2]) / imgWidth
        focalFeatures[2].append(leftMargin)

        # if a surrounding word exists, take the corresponding values to calculate; else distance is -100:
        bottomMarginRelative = -100
        if not pd.isna(featuresDF.loc[featuresDF["wordKey"] == word, "below"]).all():
            bottomMarginRelative = (int(
                featuresDF.loc[featuresDF["wordKey"] == word, "below"].item().split("_")[-1]) - int(
                word.split("_")[-1]) + int(
                featuresDF.loc[featuresDF["wordKey"] == word, "wordHeight"].item())) / imgHeight
        focalFeatures[2].append(bottomMarginRelative)

        topMarginRelative = -100
        if not pd.isna(featuresDF.loc[featuresDF["wordKey"] == word, "above"]).all():
            topMarginRelative = (int(word.split("_")[-1]) -
                                 int(featuresDF.loc[featuresDF["wordKey"] == word, "above"].item().split("_")[
                                         -1])) / imgHeight
        focalFeatures[2].append(topMarginRelative)

        rightMarginRelative = -100
        if not pd.isna(featuresDF.loc[featuresDF["wordKey"] == word, "right"]).all():
            rightMarginRelative = (int(
                featuresDF.loc[featuresDF["wordKey"] == word, "right"].item().split("_")[-2]) - int(
                word.split("_")[-2]) - int(
                featuresDF.loc[featuresDF["wordKey"] == word, "wordWidth"].item())) / imgWidth
        focalFeatures[2].append(rightMarginRelative)

        leftMarginRelative = -100
        if not pd.isna(featuresDF.loc[featuresDF["wordKey"] == word, "left"]).all():
            leftMarginRelative = (int(word.split("_")[-2]) -
                                  int(featuresDF.loc[featuresDF["wordKey"] == word, "left"].item().split("_")[
                                          -2])) / imgWidth
        focalFeatures[2].append(leftMarginRelative)

        horizontalPosition = -100
        if rightMarginRelative != -100 and leftMarginRelative != -100:
            horizontalPosition = (leftMarginRelative * imgWidth) / (
                    (leftMarginRelative + rightMarginRelative) * imgWidth)

        verticalPosition = -100
        if topMarginRelative != -100 and bottomMarginRelative != -100:
            verticalPosition = (topMarginRelative * imgHeight) / (
                    (topMarginRelative + bottomMarginRelative) * imgHeight)

        focalFeatures[2].append(horizontalPosition)
        focalFeatures[2].append(verticalPosition)

        # CATEGORICAL FEATURES

        hasDigits = "d" in textPatterns
        focalFeatures[3].append(hasDigits)

        # source for city list: https://github.com/datasets/world-cities/blob/master/scripts/process.py
        # extracted from https://download.geonames.org/export/dump/
        # contains list of cities with > 15,000 population
        isKnownCity = 0
        if citiesGazetteer is not None:
            isKnownCity = word in citiesGazetteer.values
        focalFeatures[3].append(isKnownCity)

        # source for country list: https://gist.github.com/kalinchernev/486393efcca01623b18d
        # is a list of all country names -- manually formatted Myanmar, {Burma} to easily get txt to csv
        isKnownCountry = 0
        if countryGazetteer is not None:
            isKnownCountry = word in countryGazetteer.values
        focalFeatures[3].append(isKnownCountry)

        # source for zip codes: https://download.geonames.org/export/dump/
        isKnownZip = 0
        if ZIPCodesGazetteer is not None:
            isKnownZip = word in ZIPCodesGazetteer.values
        focalFeatures[3].append(isKnownZip)

        leftAlignment = 0
        focalFeatures[2].append(leftAlignment)

        nGramLength = len(" ".join(nGram))
        focalFeatures[2].append(nGramLength)

        pageHeight = imgHeight
        focalFeatures[2].append(pageHeight)

        pageWidth = imgWidth
        focalFeatures[2].append(pageWidth)

        # start with the word itself as linesize
        lineSize = 1
        # increment lineSize as long as focalWord/focalNGram has left and right neighbour
        wordsLeft = 0
        nextWord = featuresDF.loc[featuresDF["wordKey"] == word, "left"].item()
        while not pd.isna(nextWord):
            wordsLeft += 1
            nextWord = featuresDF.loc[featuresDF["wordKey"] == nextWord, "left"].item()

        wordsRight = len(nGram) - 1
        nextWord = featuresDF.loc[featuresDF["wordKey"] == nGram[-1], "right"].item()
        while not pd.isna(nextWord):
            wordsRight += 1
            nextWord = featuresDF.loc[featuresDF["wordKey"] == nextWord, "right"].item()
        lineSize = lineSize + wordsLeft + wordsRight
        focalFeatures[2].append(lineSize)

        positionOnLine = wordsLeft / lineSize
        focalFeatures[2].append(positionOnLine)

        # Employ top-down approach: get whole area of line and subtract are of all words
        # --> for height of nGram get max height of all words enclosed within
        nGramHeight = [featuresDF.loc[featuresDF["wordKey"] == word, "wordHeight"].item() for word in nGram]
        nGramWidth = [featuresDF.loc[featuresDF["wordKey"] == word, "wordWidth"].item() for word in nGram]
        lineArea = max(nGramHeight) * imgWidth
        nGramArea = sum([h * w for h, w in zip(nGramHeight, nGramWidth)])
        lineWhiteSpace = 1 - (nGramArea / lineArea)
        focalFeatures[2].append(lineWhiteSpace)

        amountPattern = re.compile(r"^\$?\(?\-?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?\)?$")
        parsesAsAmount = bool(amountPattern.match(word.split("_")[0]))
        focalFeatures[3].append(parsesAsAmount)

        parsesAsNumber = 0
        try:
            float(word.split("_")[0])
            parsesAsNumber = 1
        except ValueError:
            pass

        # Assuming that an integer/float number cannot also be a date: primarily circumnavigates OverflowError
        # arising from too large python ints that cannot be converted to C long for date parsing
        parsesAsDate = 0
        if not parsesAsNumber:
            try:
                parse(word.split("_")[0])
                parsesAsDate = 1
            # also check for OverflowError as redundancy
            except (OverflowError, ParserError):
                pass
        focalFeatures[3].append(parsesAsDate)

        focalFeatures[3].append(parsesAsNumber)

        # instead of math.features get features from BERT_based
        isSum = featuresDF.loc[featuresDF["wordKey"] == word, "isSum"].item()
        focalFeatures[3].append(isSum)
        isMathElement = featuresDF.loc[featuresDF["wordKey"] == word, "isMathElement"].item()
        focalFeatures[3].append(isMathElement)
        isProduct = featuresDF.loc[featuresDF["wordKey"] == word, "isProduct"].item()
        focalFeatures[3].append(isProduct)

        nGramFeatures.append(focalFeatures)

    return nGramFeatures
