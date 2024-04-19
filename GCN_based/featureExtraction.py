import torch
from dateutil.parser import parse, ParserError
from dataProcessing.customDataset import CustomDataset
import pandas as pd
from PIL import Image
import numpy as np
import string


def featureCalculation(wordSeq, dataInstance):
    nGramFeatures = []

    featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
    colNames = list(featuresDF.columns)
    colNames[0] = "wordKey"
    featuresDF.columns = colNames

    imgWidth, imgHeight = Image.open(dataInstance["pathInvoicePNG"]).size
    wordSeq = wordSeq.split()

    for word in featuresDF.loc[:, "wordKey"].tolist():
        if word.split("_")[0] in string.punctuation:
            continue
        focalFeatures = [word, []]

        # Boolean features

        parsesAsDate = 0
        try:
            parse(word.split("_")[0])
            parsesAsDate = 1
        except ParserError:
            pass
        focalFeatures[1].append(parsesAsDate)

        isZipCode = 0
        focalFeatures[1].append(isZipCode)

        isKnownCity = 0
        focalFeatures[1].append(isKnownCity)
        isKnownDep = 0
        focalFeatures[1].append(isKnownDep)
        isKnownCountry = 0
        focalFeatures[1].append(isKnownCountry)

        isAlphabetic = 0
        focalFeatures[1].append(isAlphabetic)
        isNumeric = 0
        focalFeatures[1].append(isNumeric)
        isAlphaNumeric = 0
        focalFeatures[1].append(isAlphaNumeric)
        isNumberWithDecimal = 0
        focalFeatures[1].append(isNumberWithDecimal)
        isRealNumber = 0
        focalFeatures[1].append(isRealNumber)
        isCurrency = 0
        focalFeatures[1].append(isCurrency)
        hasRealAndCurrencySign = 0
        focalFeatures[1].append(hasRealAndCurrencySign)
        hasRealAndCurrencyWord = 0
        focalFeatures[1].append(hasRealAndCurrencyWord)

        # Numeric features
        bottomMarginRelative = -100
        if featuresDF.loc[featuresDF["wordKey"] == word, "below"].item() is not np.nan:
            bottomMarginRelative = (int(
                featuresDF.loc[featuresDF["wordKey"] == word, "below"].item().split("_")[2]) - int(
                word.split("_")[2]) + int(
                featuresDF.loc[featuresDF["wordKey"] == word, "wordHeight"].item())) / imgHeight
        focalFeatures[1].append(bottomMarginRelative)

        topMarginRelative = -100
        if featuresDF.loc[featuresDF["wordKey"] == word, "above"].item() is not np.nan:
            topMarginRelative = (int(word.split("_")[2]) -
                                 int(featuresDF.loc[featuresDF["wordKey"] == word, "above"].item().split("_")[
                                         2])) / imgHeight
        focalFeatures[1].append(topMarginRelative)

        rightMarginRelative = -100
        if featuresDF.loc[featuresDF["wordKey"] == word, "right"].item() is not np.nan:
            rightMarginRelative = (int(
                featuresDF.loc[featuresDF["wordKey"] == word, "right"].item().split("_")[1]) - int(
                word.split("_")[1]) - int(
                featuresDF.loc[featuresDF["wordKey"] == word, "wordWidth"].item())) / imgWidth
        focalFeatures[1].append(rightMarginRelative)

        leftMarginRelative = -100
        if featuresDF.loc[featuresDF["wordKey"] == word, "left"].item() is not np.nan:
            leftMarginRelative = (int(word.split("_")[1]) -
                                  int(featuresDF.loc[featuresDF["wordKey"] == word, "left"].item().split("_")[
                                          1])) / imgWidth
        focalFeatures[1].append(leftMarginRelative)

        nGramFeatures.append(focalFeatures)

    return nGramFeatures


if __name__ == '__main__':
    pass