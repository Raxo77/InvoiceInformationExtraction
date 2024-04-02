from torch.utils.data import Dataset
import os
from bs4 import BeautifulSoup
from dataProcessing.filterRawDataset import listDirectory
from utils.CONFIG_PATH import CONFIG_PATH
from utils.helperFunctions import getConfig, loadJSON
import dataProcessing.OCR as OCR
from dataProcessing.filterRawDataset import getGoldLabels
import warnings

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
        #print(instanceFolderPath)
        instanceFolderContent = listDirectory(instanceFolderPath, folderOnly=False)

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
                imageList = OCR.convertPDFtoImage(pdfFile.path)
                hOCR = OCR.OCRengine(imageList,
                                     saveResultsPath=f"{instanceFolderPath}\\hOCR_output.xml")[0]

        try:
            wordposFeatures = loadJSON(f"{instanceFolderPath}\\wordposFeatures.json")
        except FileNotFoundError:
            wordposFeatures = None

        try:
            zonesegFeatures = loadJSON(f"{instanceFolderPath}\\zonesegFeatures.json")
        except FileNotFoundError:
            zonesegFeatures = None

        try:
            goldLabels = loadJSON(os.path.join(instanceFolderPath, "goldLabels.json"))
        except FileNotFoundError:
            getGoldLabels(os.path.join(instanceFolderPath, "ground_truth_tags.json"),
                          getConfig("targetLabels", CONFIG_PATH),
                          instanceFolderPath)
            goldLabels = loadJSON(os.path.join(instanceFolderPath, "goldLabels.json"))

        data = {"instanceFolderPath": instanceFolderPath,
                "pathInvoicePDF": os.path.join(instanceFolderPath, "flat_document.pdf"),
                "pathInvoicePNG": os.path.join(instanceFolderPath, "flat_document.png"),
                "hOCR": BeautifulSoup(hOCR, features="lxml"),
                "zonesegFeatures": zonesegFeatures,
                "wordposFeatures": wordposFeatures,
                "BERT-basedFeaturesPath": os.path.join(instanceFolderPath, "BERT_features_noPunct.csv"),
                "goldLabels": goldLabels
                }

        return data

data = CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
#print(data.__getitem__(2))

# data = CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
# data.__getitem__(0)
# print(data.__getitem__(0))
