import sys
import os
from helperFunctions import getConfig

from helperFunctions import createJSON, loadJSON, separate
from pdf2image import convert_from_path




def imageIsBlank():
    pass


def convertPDFtoImage(pathToPDF):
    # include check whether invoice page/image is blank page

    images = convert_from_path(pathToPDF, fmt="png", output_folder=r"C:\Users\fabia\NER_for_IIE\data\00001\\")


def OCRengine():
    pass


def compareOCRwithGroundTruth():
    pass


# convertPDFtoImage(r"C:\Users\fabia\NER_for_IIE\data\00001\flat_document.pdf")
DATA_PATH = getConfig("pathToDataFolder", "configDataProcessing.json")
a = (list(os.scandir(DATA_PATH)))

