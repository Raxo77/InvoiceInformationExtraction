from utils.helperFunctions import getConfig, separate, loadJSON, createJSON
import os
from utils.CONFIG_PATH import CONFIG_PATH

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
All the other ones will be automatically deleted, drastically reducing dataset size
In a second step, pdf will be fed to an OCR engine. The OCR output will then be
compared to the ground truth labels to assess OCR accuracy and perform a ceiling analysis.
"""

CONFIG = CONFIG_PATH
PATH_TO_DATA_FOLDER = getConfig("pathToDataFolder", CONFIG)
FILES_TO_KEEP = getConfig("filesToKeep", CONFIG)


def listDirectory(path=PATH_TO_DATA_FOLDER, folderOnly=True) -> list:
    dirList = list(os.scandir(path))

    if folderOnly:
        return [i for i in dirList if i.is_dir()]

    else:
        return dirList


def filterFiles(dirList, whitelist=FILES_TO_KEEP, feedback=False):
    for directory in dirList:
        for file in listDirectory(directory.path, folderOnly=False):
            if file.name not in whitelist:
                os.remove(file.path)
                if feedback:
                    print(f"removed {file.name} from {file.path}")
        if feedback:
            separate()


def getGoldLabels(pathToJSON: str, targets: dict, dirPath=""):
    # assuming JSON is structured like ground_truth_tags.json
    data = loadJSON(pathToJSON)

    data = {
        entry["tag"]: {k: v for k, v in entry.items() if k != "tag"}
        for entry in data
    }
    goldLabels = {i: [] for i in targets.keys()}
    for k, v in targets.items():
        for i in v:
            try:
                goldLabels[k].append(data[i])
            except KeyError:
                goldLabels[k].append(None)
    temp = goldLabels
    goldLabels = {}
    for k,v in temp.items():
        if len(v) > 0:
            goldLabels[k] = v[0]
        else: goldLabels[k] = None
    #goldLabels = {k: v[0] for k, v in goldLabels.items() if len(v) > 0 else k: None}

    if dirPath:
        createJSON(f"{dirPath}\\goldLabels.json", goldLabels)

    return goldLabels

# RUN FOR ALL:
# filterFiles(listDirectory(),feedback=True)
