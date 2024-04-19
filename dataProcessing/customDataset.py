import pandas as pd
from torch.utils.data import Dataset
import os
from bs4 import BeautifulSoup
from dataProcessing.filterRawDataset import listDirectory
from utils.helperFunctions import getConfig, loadJSON, CONFIG_PATH
import dataProcessing.OCR as OCR
import TemplateDetection.templateDetection as TemplateDetection
from dataProcessing.filterRawDataset import getGoldLabels
import warnings
from PIL import Image, ImageDraw, ImageFont
import BERT_based.featureExtraction as BERTbased_featureExtraction

warnings.filterwarnings("ignore", category=UserWarning, message=".*It looks like you're parsing an XML document.*")


class CustomDataset(Dataset):
    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.instances = listDirectory(rootDir)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, instanceIdx):
        """
        :param instanceIdx:
        :return: dict with hOCR result and information to derive further features if desired
        """

        instanceFolderPath = self.instances[instanceIdx].path
        instanceFolderContent = listDirectory(instanceFolderPath, folderOnly=False)

        data = {"instanceFolderPath": instanceFolderPath,
                "pathInvoicePDF": os.path.join(instanceFolderPath, "flat_document.pdf"),
                "pathInvoicePNG": os.path.join(instanceFolderPath, "flat_document.png"),
                "hOCR": None,
                "zonesegFeatures": None,
                "wordposFeatures": None,
                "BERT-basedFeaturesPath": os.path.join(instanceFolderPath, "BERT_features.csv"),
                "BERT-basedNoPunctuationFeaturesPath": os.path.join(instanceFolderPath, "BERT_features_noPunct.csv"),
                "goldLabels": None
                }

        # Get OCR and load into data dict
        try:
            with open(os.path.join(instanceFolderPath, "hOCR_output.xml"), 'r', encoding='utf-8') as f:
                hOCR = f.read()
        except FileNotFoundError:
            pdfFile = []
            for file in instanceFolderContent:
                if file.name.endswith(".pdf"):
                    pdfFile.append(file)
            if len(pdfFile) == 1:
                pdfFile = pdfFile[0]
                imageList = OCR.convertPDFtoImage(pdfFile.path, getConfig("imageIsBlankThreshold", CONFIG_PATH))
                hOCR = OCR.OCRengine(imageList,
                                     saveResultsPath=f"{instanceFolderPath}\\hOCR_output.xml")[0]
        data["hOCR"] = BeautifulSoup(hOCR, features="lxml")

        # Get wordpos features and load into data dict
        try:
            print(f"{instanceFolderPath}\\wordposFeatures.json")
            wordposFeatures = loadJSON(f"{instanceFolderPath}\\wordposFeatures.json")
        except FileNotFoundError:
            wordposFeatures = TemplateDetection.wordposFeatures(data)

        # Get zoneseg features and load into data dict
        try:
            zonesegFeatures = loadJSON(f"{instanceFolderPath}\\zonesegFeatures.json")
        except FileNotFoundError:
            zonesegFeatures = TemplateDetection.zonesegFeatures(data)

        # Get gold labels and load into data dict
        try:
            goldLabels = loadJSON(os.path.join(instanceFolderPath, "goldLabels.json"))
        except FileNotFoundError:
            getGoldLabels(os.path.join(instanceFolderPath, "ground_truth_tags.json"),
                          getConfig("targetLabels", CONFIG_PATH),
                          instanceFolderPath)
            goldLabels = loadJSON(os.path.join(instanceFolderPath, "goldLabels.json"))

        try:
            pd.read_csv(os.path.join(instanceFolderPath, "BERT_features.csv"))
        except FileNotFoundError:
            BERTbased_featureExtraction.deriveFeatures(data, includePunct=True)
        try:
            pd.read_csv(os.path.join(instanceFolderPath, "BERT_features_noPunct.csv"))
        except FileNotFoundError:
            BERTbased_featureExtraction.deriveFeatures(data, includePunct=False)

        data = {"instanceFolderPath": instanceFolderPath,
                "pathInvoicePDF": os.path.join(instanceFolderPath, "flat_document.pdf"),
                "pathInvoicePNG": os.path.join(instanceFolderPath, "flat_document.png"),
                "hOCR": BeautifulSoup(hOCR, features="lxml"),
                "zonesegFeatures": zonesegFeatures,
                "wordposFeatures": wordposFeatures,
                "BERT-basedFeaturesPath": os.path.join(instanceFolderPath, "BERT_features.csv"),
                "BERT-basedNoPunctFeaturesPath": os.path.join(instanceFolderPath, "BERT_features_noPunct.csv"),
                "goldLabels": goldLabels
                }

        return data

    def plotInstance(self, instanceIdx: int, saveImage=False, highlightGoldLabels=True):
        dataInstance = self.__getitem__(instanceIdx)

        image = Image.open(dataInstance["pathInvoicePNG"])
        draw = ImageDraw.Draw(image)

        words = dataInstance["hOCR"].find_all("span", class_="ocrx_word")
        for word in words:
            coords = word["title"].split(";")[0].split(" ")
            x1, y1, x2, y2 = map(int, [coords[1], coords[2], coords[3], coords[4]])
            draw.rectangle([x1, y1, x2, y2], outline='lightgreen', width=3)

        if highlightGoldLabels:
            for label in dataInstance["goldLabels"].keys():
                if dataInstance["goldLabels"][label] is not None:
                    _, y1, x1, height, width = list(dataInstance["goldLabels"][label].values())
                    x2 = x1 + width
                    y2 = y1 + height
                    draw.rectangle([x1, y1, x2, y2], outline='gold', width=3)
                    draw.text((x1, y1 - 32), label, fill="black", font=ImageFont.truetype("arial.ttf", 32))

        if saveImage:
            image.save(os.path.join(dataInstance["instanceFolderPath"], "boundingBoxImage.png"))

        image.show()


if __name__ == '__main__':
    data = CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
    print(data[0])
    data.plotInstance(0)
# print(data.__getitem__(2))

# data = CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
# data.__getitem__(0)
# print(data.__getitem__(0))
