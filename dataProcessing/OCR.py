import os
import pytesseract
import Levenshtein
from bs4 import BeautifulSoup
from PIL import Image, ImageStat
from pdf2image import convert_from_path
from dataProcessing.filterRawDataset import listDirectory
from utils.helperFunctions import getConfig, loadJSON, CONFIG_PATH

CONFIG = CONFIG_PATH
GROUND_TRUTH_FILE_NAME = getConfig("groundTruthFileName", CONFIG)
PATH_TO_DATA_FOLDER = getConfig("pathToDataFolder", CONFIG)
IMAGE_IS_BLANK_THRESHOLD = getConfig("imageIsBlankThreshold", CONFIG)


def convertPDFtoImage(pathToPDF: str, imageIsBlankThreshold: int, outputFolder: str = "") -> list:
    images = convert_from_path(pathToPDF)
    imageList = []

    for count, image in enumerate(images):
        if not imageIsBlank(image, imageIsBlankThreshold):

            if outputFolder == "<same>":
                imagePath = f"{os.path.split(pathToPDF)[0]}\\invoiceImage_{count}.png"


            elif outputFolder:
                imagePath = f"{outputFolder}\\invoiceImage_{count}.png"
                image.save(imagePath)
                imageList.append(imagePath)


            imageList.append(image)

    return imageList


def imageIsBlank(image, threshold: int):
    return ImageStat.Stat(image.convert("L")).mean[0] >= threshold


def OCRengine(imageList: list, saveResultsPath="", groundTruthPath=""):
    hOCR_list = []

    for imageInfo in imageList:
        if imageList[0] is str:
            image = Image.open(imageInfo)
        else:
            image = imageInfo

        hOCR_data = pytesseract.image_to_pdf_or_hocr(image, extension="hocr")
        hOCR_list.append(hOCR_data.decode("utf-8"))

    hOCR_xml = BeautifulSoup(hOCR_list[0], features="lxml")

    for pageCount, result in enumerate(hOCR_list[1:]):
        temp = BeautifulSoup(result.replace(f"page_1", f"page_{pageCount + 2}"), features="lxml")
        temp = temp.find("body").contents
        hOCR_xml.find("body").extend(temp)
    hOCR_xml = str(hOCR_xml)

    if saveResultsPath:
        with open(saveResultsPath, "w", encoding="utf-8") as f:
            f.write(hOCR_xml)

    if groundTruthPath:
        temp = compareOCRwithGroundTruth(hOCR_xml, groundTruthPath)
        return hOCR_xml, temp

    else:
        return hOCR_xml, None


def compareOCRwithGroundTruth(hOCR_output, groundTruthPath):
    """
    Calculate similarity of OCRed words with ground truth text via Levenshtein distance;
    Respective strings are first sorted alphabetically to neutralise the potentially
    different order in which words are extracted/stored;
    Also replace whitespaces to neutralise any differences that might occur due to
    different tokenization of OCR output and ground truth text
    """

    parsedOutput = BeautifulSoup(hOCR_output, features="lxml")
    words = parsedOutput.find_all("span", attrs="ocrx_word")
    words = "".join(sorted(word.get_text() for word in words))

    groundTruthText = loadJSON(groundTruthPath)
    groundTruthText = "".join(sorted(i["text"] for i in groundTruthText))

    distance = Levenshtein.distance(words, groundTruthText)
    similarity = 1 - distance / max(len(words), len(groundTruthText))

    return similarity


def runForAll(imageIsBlankThreshold=IMAGE_IS_BLANK_THRESHOLD):
    for directory in listDirectory(PATH_TO_DATA_FOLDER):
        pdfFile = []
        for file in listDirectory(directory.path, folderOnly=False):
            if file.name.endswith("pdf"):
                pdfFile.append(file)
        if len(pdfFile) == 1:
            pdfFile = pdfFile[0]
            imageList = convertPDFtoImage(pdfFile.path, imageIsBlankThreshold)
            OCRengine(imageList,
                      f"{directory.path}\\hOCR_output.xml",
                      f"{directory.path}\\{GROUND_TRUTH_FILE_NAME}")
        else:
            print(directory.name)

