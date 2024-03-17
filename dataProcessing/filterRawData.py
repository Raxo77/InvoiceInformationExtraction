import sys

from helperFunctions import createJSON, loadJSON, separate
from pdf2image import convert_from_path

"""
  Of the files provided per invoice, we need:
* flat_document.pdf
* flat_document.png
* details.json
* ground_truth_structure.json
* ground_truth_tags.json
* ground_truth_words.json
  (And maybe as nice to have:
* flat_document.png
* flat_information_delta.png
* flat_template.png
* flat_text_mask.png
"""

def imageIsBlank():
    pass


def convertPDFtoImage(pathToPDF):
    # include check whether invoice page/image is blank page

    images = convert_from_path(pathToPDF,fmt="png", output_folder=r"C:\Users\fabia\NER_for_IIE\data\00001\\")


def OCRengine():
    pass


def compareOCRwithGroundTruth():
    pass

convertPDFtoImage(r"C:\Users\fabia\NER_for_IIE\data\00001\flat_document.pdf")
