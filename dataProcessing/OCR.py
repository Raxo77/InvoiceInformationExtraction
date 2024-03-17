from utils.helperFunctions import getConfig, separate, loadJSON
from filterRawDataset import listDirectory, CONFIG
from pdf2image import convert_from_path
from PIL import Image, ImageStat
import pytesseract
import Levenshtein as lev
from bs4 import BeautifulSoup


def convertPDFtoImage(pathToPDF: str, outputFolder=""):
    images = convert_from_path(pathToPDF)
    imageList = []

    for count, image in enumerate(images):
        if not False:  # imageIsBlank(image):

            if outputFolder:
                imagePath = f"{outputFolder}\\invoiceImage_{count}.png"
                image.save(imagePath)
                imageList.append(imagePath)

            else:
                imageList.append(image)

    return imageList


def imageIsBlank(image, threshold=254):
    return ImageStat.Stat(image.convert("L")).mean[0] >= threshold


def OCRengine(imageList: list, saveResultsPath=""):
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

    return hOCR_xml


def compareOCRwithGroundTruth(hOCR_output, groundTruthPath):
    parsedOutput = BeautifulSoup(hOCR_output, features="lxml")
    words = parsedOutput.find_all("span", attrs="ocrx_word")
    words = " ".join(word.get_text() for word in words)

    groundTruthText = loadJSON(groundTruthPath)
    groundTruthText = " ".join(i["text"] for i in groundTruthText)

    missingCount = [0., []]
    temp = groundTruthText.split()
    for word in words.split():
        if word not in temp:
            if word[:-1] in temp:
                missingCount[0] += .5
                missingCount[1].append(word[:-1])
            else:
                missingCount[0] += 1
                missingCount[1].append(word)

    distance = lev.distance(words, groundTruthText)
    similarity = 1 - distance / max(len(words), len(groundTruthText))

    return similarity, missingCount


images = convertPDFtoImage(r"C:\Users\fabia\NER_for_IIE\data\00001\flat_document.pdf")
a = OCRengine(images)
b = compareOCRwithGroundTruth(a, r"C:\Users\fabia\NER_for_IIE\data\00001\ground_truth_words.json")
print(b)
