from utils.helperFunctions import getConfig, loadJSON
from dataProcessing.filterRawDataset import listDirectory
from pdf2image import convert_from_path
from PIL import Image, ImageStat
import pytesseract
import Levenshtein
from bs4 import BeautifulSoup
from utils.CONFIG_PATH import CONFIG_PATH

CONFIG = CONFIG_PATH
GROUND_TRUTH_FILE_NAME = getConfig("groundTruthFileName", CONFIG)
PATH_TO_DATA_FOLDER = getConfig("pathToDataFolder", CONFIG)



def convertPDFtoImage(pathToPDF: str, outputFolder=""):
    images = convert_from_path(pathToPDF)
    imageList = []

    for count, image in enumerate(images):
        if not imageIsBlank(image):

            if outputFolder:
                imagePath = f"{outputFolder}\\invoiceImage_{count}.png"
                image.save(imagePath)
                imageList.append(imagePath)

            else:
                imageList.append(image)

    return imageList


def imageIsBlank(image, threshold=254.5):
    return ImageStat.Stat(image.convert("L")).mean[0] >= threshold


def OCRengine(imageList: list, saveResultsPath="", groundTruthPath=""):
    hOCR_list = []

    for imageInfo in imageList:
        if imageList[0] is str:
            image = Image.open(imageInfo)
        else:
            image = imageInfo

        hOCR_data = pytesseract.image_to_pdf_or_hocr(image, extension="hocr")
        print(hOCR_data)
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
        return hOCR_xml, compareOCRwithGroundTruth(hOCR_xml, groundTruthPath)

    else:
        return hOCR_xml, None


def compareOCRwithGroundTruth(hOCR_output, groundTruthPath):
    parsedOutput = BeautifulSoup(hOCR_output, features="lxml")
    words = parsedOutput.find_all("span", attrs="ocrx_word")
    words = " ".join(word.get_text() for word in words)

    groundTruthText = loadJSON(groundTruthPath)
    groundTruthText = " ".join(i["text"] for i in groundTruthText)

    missingCount = [0., []]
    temp = groundTruthText.split()
    temp = [i.replace(",", "") for i in temp]
    for word in words.split():
        if word.replace(",", "") not in temp:
            missingCount[0] += 1
            missingCount[1].append(word)

    distance = Levenshtein.distance(words, groundTruthText)
    similarity = 1 - distance / max(len(words), len(groundTruthText))

    return similarity, missingCount


def runForAll():
    for directory in listDirectory(PATH_TO_DATA_FOLDER):
        pdfFile = []
        for file in listDirectory(directory.path, folderOnly=False):
            if file.name.endswith("pdf"):
                pdfFile.append(file)
        if len(pdfFile) == 1:
            pdfFile = pdfFile[0]
            imageList = convertPDFtoImage(pdfFile.path)
            OCRengine(imageList,
                      f"{directory.path}\\hOCR_output.xml",
                      f"{directory.path}\\{GROUND_TRUTH_FILE_NAME}")
        else:
            print(directory.name)


# imgList = convertPDFtoImage(r"C:\Users\fabia\Downloads\test.pdf")
# OCRengine(imgList)
