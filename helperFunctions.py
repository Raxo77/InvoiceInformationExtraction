import os
import json


def separate(x="-"):
    print(x * 50)


def loadJSON(pathToData: str):
    with open(pathToData, "r") as f:
        data = json.load(f)

    return data


def createJSON(pathNewJSON, content):
    with open(pathNewJSON, "w") as outfile:
        json.dump(content, outfile)


def getConfig(entity: str, pathToConfig: str):
    return loadJSON(pathToConfig)[entity]
