import re
import torch
import string
import numpy as np
import pandas as pd
from PIL import Image
from dateutil.parser import parse, ParserError
from dataProcessing.customDataset import CustomDataset


def checkWhetherIsNumberWithDecimal(inputString: str) -> bool:
    if "," in inputString or "." in inputString:
        inputString = inputString.translate(str.maketrans({",": "", ".": "", " ": ""}))
        try:
            float(inputString)
            return True
        except ValueError:
            return False
    return False


def checkWhetherIsRealNumer(inputString: str) -> bool:
    # The focal string is considered a real number either if it resembles a fraction
    # or can be parsed to a float without any pre-processing (e.g. removing "." or ",")

    # Compile regex pattern to matches positive and negative fractions
    fractionPattern = re.compile(r'^-?\+?\d+/\d+$')

    # check if string directly resembles a fraction
    if fractionPattern.match(inputString):
        return True

    # check if string can be parsed to float
    try:
        float(inputString)
        return True
    except ValueError:
        return False


def featureCalculation(wordSeq,
                       dataInstance,
                       citiesGazetteer=None,
                       countryGazetteer=None,
                       ZIPCodesGazetteer=None
                       ):
    """
    Derive the custom features for the GCN-based model as they are described in the respective paper
    "An Invoice Reading System Using a Graph Convolutional Network"
    """

    featuresDF = pd.read_csv(dataInstance["BERT-basedNoPunctFeaturesPath"])

    nGramFeatures = []

    imgWidth, imgHeight = Image.open(dataInstance["pathInvoicePNG"]).size

    for word in featuresDF.loc[:, "wordKey"].tolist():

        focalFeatures = [word, []]

        # BOOLEAN FEATURES

        parsesAsDate = 0
        try:
            parse(word.split("_")[0])
            parsesAsDate = 1
        except (OverflowError, ParserError):
            pass
        focalFeatures[1].append(parsesAsDate)

        isZipCode = word in ZIPCodesGazetteer.values if ZIPCodesGazetteer is not None else 0
        focalFeatures[1].append(isZipCode)

        isKnownCity = word in citiesGazetteer.values if citiesGazetteer is not None else 0
        focalFeatures[1].append(isKnownCity)

        isKnownCountry = word in countryGazetteer.values if citiesGazetteer is not None else 0
        focalFeatures[1].append(isKnownCountry)

        isAlphabetic = word.isalpha()
        focalFeatures[1].append(isAlphabetic)

        isNumeric = word.isnumeric()
        focalFeatures[1].append(isNumeric)

        isAlphaNumeric = word.isalnum()
        focalFeatures[1].append(isAlphaNumeric)

        isNumberWithDecimal = checkWhetherIsNumberWithDecimal(word)
        focalFeatures[1].append(isNumberWithDecimal)

        isRealNumber = checkWhetherIsRealNumer(word)
        focalFeatures[1].append(isRealNumber)

        # Compile a regex pattern to match strings resembling monetary amounts
        # Notably, this pattern currently only recognises € and $ as currency designations and
        # may as such be refined depending on the concrete dataset at han
        # On the same note, given that all punctuation characters have been removed for these means of
        # feature extraction, checking for "," or "." may be redundant, does, however, allow for more flexibility
        currencyPattern = re.compile(r'^\$?€?\s?-?(\d{1,3}(,\d{3})*|\d+)(\.\d{1,2})?$')
        isCurrency = bool(currencyPattern.match(word.strip()))
        focalFeatures[1].append(isCurrency)

        # NUMERIC FEATURES

        bottomMarginRelative = -100
        if not pd.isna(featuresDF.loc[featuresDF["wordKey"] == word, "below"].item()):
            bottomMarginRelative = (int(
                featuresDF.loc[featuresDF["wordKey"] == word, "below"].item().split("_")[2]) - int(
                word.split("_")[2]) + int(
                featuresDF.loc[featuresDF["wordKey"] == word, "wordHeight"].item())) / imgHeight
        focalFeatures[1].append(bottomMarginRelative)

        topMarginRelative = -100
        if not pd.isna(featuresDF.loc[featuresDF["wordKey"] == word, "above"].item()):
            topMarginRelative = (int(word.split("_")[2]) -
                                 int(featuresDF.loc[featuresDF["wordKey"] == word, "above"].item().split("_")[
                                         2])) / imgHeight
        focalFeatures[1].append(topMarginRelative)

        rightMarginRelative = -100
        if not pd.isna(featuresDF.loc[featuresDF["wordKey"] == word, "right"].item()):
            rightMarginRelative = (int(
                featuresDF.loc[featuresDF["wordKey"] == word, "right"].item().split("_")[1]) - int(
                word.split("_")[1]) - int(
                featuresDF.loc[featuresDF["wordKey"] == word, "wordWidth"].item())) / imgWidth
        focalFeatures[1].append(rightMarginRelative)

        leftMarginRelative = -100
        if not pd.isna(featuresDF.loc[featuresDF["wordKey"] == word, "left"].item()):
            leftMarginRelative = (int(word.split("_")[1]) -
                                  int(featuresDF.loc[featuresDF["wordKey"] == word, "left"].item().split("_")[
                                          1])) / imgWidth
        focalFeatures[1].append(leftMarginRelative)

        nGramFeatures.append(focalFeatures)

    return nGramFeatures
