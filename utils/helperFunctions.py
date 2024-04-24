import json
import string
import pandas as pd

def separate(x="-", num=50, sep=" ", end="\n"):
    print(x * num, sep=sep, end=end)


def loadJSON(pathToData: str):
    with open(pathToData, "r") as f:
        data = json.load(f)

    return data


def createJSON(pathNewJSON: str, content: dict):
    with open(pathNewJSON, "w") as outfile:
        json.dump(content, outfile)


def getConfig(entity: str, pathToConfig: str):
    return loadJSON(pathToConfig)[entity]


def flattenList(nestedList):
    flatList = []
    for element in nestedList:
        if isinstance(element, list):
            flatList.extend(flattenList(element))
        else:
            flatList.append(element)
    return flatList







# Path to the global config file for all extractionScripts, models and the ensemble
# - will be imported by all subsequent modules
# CONFIG_PATH = r"C:\Users\fabia\NER_for_IIE\utils\configGlobal.json"
CONFIG_PATH = r"C:\Users\fabia\InvoiceInformationExtraction\utils\configGlobalDESKTOP.json"
