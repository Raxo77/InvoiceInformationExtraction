import pandas as pd
from datetime import datetime
from dataProcessing.customDataset import loadJSON, CustomDataset
from utils.helperFunctions import createJSON, getConfig, CONFIG_PATH
from TemplateDetection.templateDetection import wordposSimilarity, zonesegSimilarity, plotGrid


def bboxOverlap(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Check if there is no overlap
    if x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max:
        return False

    # Otherwise, the boxes overlap
    return True

    # Ellipsis approach
    center1 = calcBboxCenter(bbox1)
    center2 = calcBboxCenter(bbox2)
    r = 1.1
    semiMajorAxis = abs(x1_min - x1_max) / 2 * r
    semiMinorAxis = abs(y1_max - y1_min) / 2 * r

    return checkIfInsideEllipsis(center2, center1, semiMajorAxis, semiMinorAxis)


def calcBboxCenter(bbox):
    x1, y1, x2, y2 = bbox
    centerX = (x1 + x2) / 2
    centerY = (y1 + y2) / 2
    return centerX, centerY


def checkIfInsideEllipsis(point, center, semiMajorAxis, semiMinorAxis):
    px, py = point
    cx, cy = center
    return ((px - cx) ** 2) / semiMajorAxis ** 2 + ((py - cy) ** 2) / semiMinorAxis ** 2 <= 1


class TemplateBasedExtraction:

    def __init__(self,
                 pathToPeers: str,
                 pathToScripts: str,
                 referenceDataset
                 ):

        self.peerGroupsPath = pathToPeers
        self.peerGroups = pd.read_csv(pathToPeers, dtype={"itemNumber": str})
        self.extractionScripts = loadJSON(pathToScripts)
        self.referenceData = referenceDataset

    def compareZoneseg(self, dataInstance):
        peersDict = self.peerGroups.groupby("templateNumber")["itemNumber"].apply(list).to_dict()

        avgSimZoneseg = {k: 0 for k in peersDict.keys()}
        for k in peersDict.keys():
            similarityList = []
            for sampleNum in peersDict[k]:
                referenceIndex = self.peerGroups.loc[
                    self.peerGroups["itemNumber"] == sampleNum, "itemIndexDataset"].item()
                referenceSample = self.referenceData[referenceIndex]
                similarityList.append(zonesegSimilarity(zoneseg1=dataInstance["zonesegFeatures"]["zonesegFeatures"],
                                                        zoneseg2=referenceSample["zonesegFeatures"]["zonesegFeatures"]))
            avgSimZoneseg[k] = sum(similarityList) / len(similarityList)

        if max(list(avgSimZoneseg.values())) <= -3.8:
            # if all similarity score should be smaller than -4, assume no pre-defined template exists
            return -1
        else:
            # else return the three most similar templates
            return sorted(avgSimZoneseg, key=avgSimZoneseg.get, reverse=True)[:3]

    def compareWordpos(self, dataInstance):
        peersDict = self.peerGroups.groupby("templateNumber")["itemNumber"].apply(list).to_dict()

        avgSimWordpos = {k: 0 for k in peersDict.keys()}
        for k in peersDict.keys():
            similarityList = []
            for sampleNum in peersDict[k]:
                referenceIndex = self.peerGroups.loc[
                    self.peerGroups["itemNumber"] == sampleNum, "itemIndexDataset"].item()
                referenceSample = self.referenceData[referenceIndex]
                similarityList.append(wordposSimilarity(wordpos1=dataInstance["wordposFeatures"]["wordposFeatures"],
                                                        wordpos2=referenceSample["wordposFeatures"]["wordposFeatures"]))
            avgSimWordpos[k] = sum(similarityList) / len(similarityList)
        if max(list(avgSimWordpos.values())) <= -85:
            # if all similarity score should be smaller than -4, assume no pre-defined template exists
            return -1
        else:
            # else return the three most similar templates
            return sorted(avgSimWordpos, key=avgSimWordpos.get, reverse=True)[:3]

    def findTemplate(self, dataInstance, activateSemiManualLabelling=False):
        """

        :param dataInstance:
        :param activateSemiManualLabelling: if an integer is passed, the int is considered the index of the current
        data instance within the reference dataset. The user will be asked if the found template should be added to
        peerGroup.csv - the user will have the option to alter the found template before addition
        :return:
        """
        templateZoneseg = self.compareZoneseg(dataInstance)
        templateWordpos = self.compareWordpos(dataInstance)

        """
        Finding the final template will be done as follows:
        ---------------------------------------------------
        1) If both approaches dont return any templates --> consider template not found
        2) If only one approach returns a template --> first template
        3) If both approaches return identical templates --> common template
        4) If both approaches return dissimilar templates:
            4.1) If respective top 2 are identical --> choose template randomly between these two
            4.2) Else --> consider template not found
                 (|--> using only the top 2 found templates is predominantly owed to the currently relatively small
                       number of pre-defined templates; may be increased if labelled template number increases)
        """

        foundTemplate = -1

        # 1)
        if templateZoneseg == -1 and templateWordpos == -1:
            foundTemplate = -1

        # 2)
        elif templateZoneseg == -1 or templateWordpos == -1:
            if templateZoneseg == -1:
                foundTemplate = templateWordpos[0]
            else:
                foundTemplate = templateZoneseg[0]

        # 3)
        elif templateZoneseg[0] == templateWordpos[0]:
            foundTemplate = templateZoneseg[0]

        # 4)
        else:
            # 4.1)
            if set(templateZoneseg[:2]) == set(templateWordpos[:2]):
                foundTemplate = set(templateZoneseg)[0]
            # 4.2)
            else:
                foundTemplate = -1

        if type(activateSemiManualLabelling) == int:
            name = dataInstance["instanceFolderPath"].split("\\")[-1]
            if name not in self.peerGroups.itemNumber.tolist():
                print(f"foundTemplate for {name}: {foundTemplate}")
                decision = input("Add foundTemplate to peerGroups.csv (y/n or new templateNumber)?\n")
                if decision.lower() == "y":
                    toAdd = [activateSemiManualLabelling, name, foundTemplate, "detected"]
                elif decision.lower() == "n":
                    return foundTemplate
                else:
                    toAdd = [activateSemiManualLabelling, name, decision, "detected"]

                self.peerGroups.loc[len(self.peerGroups)] = toAdd
                self.peerGroups.to_csv(self.peerGroupsPath, index=False)
                print("Added successfully")

        return foundTemplate

    def findEntities(self, dataInstance, *args):
        foundTemplate = self.findTemplate(dataInstance, *args)

        if foundTemplate == -1 or foundTemplate not in self.extractionScripts.keys():
            return {"foundTemplate": foundTemplate}

        foundEntities = {}
        extractionScript = self.extractionScripts[foundTemplate]

        # for each entity contained in the document as per gold labels:
        for entity in extractionScript:
            foundEntities[entity] = []
            """
            Coordinates of the bounding boxes in hOCR output correspond to
            x_topLeft, y_topLeft, x_bottomRight, y_bottomRight
            As such, those coordinates need to be derived from the extraction script
            """
            y1_gold, x1_gold, y2_gold, x2_gold = extractionScript[entity].values()
            x2_gold += x1_gold
            y2_gold += y1_gold

            # for each word extracted by the hOCR engine:
            hOCR = dataInstance["hOCR"]
            for span in hOCR.find_all("span", class_="ocrx_word"):
                title = span.get("title")
                if title:
                    parts = title.split(";")
                    parts = [part for part in parts if part.strip().startswith("bbox")][0]
                    _, x1_span, y1_span, x2_span, y2_span = parts.strip().split()
                    x1_span, y1_span, x2_span, y2_span = int(x1_span), int(y1_span), int(x2_span), int(y2_span)
                    if bboxOverlap((x1_gold, y1_gold, x2_gold, y2_gold), (x1_span, y1_span, x2_span, y2_span)):
                        foundEntities[entity].append(span.get_text())
        foundEntities = self.cleanEntitiesDict(foundEntities)
        foundEntities["foundTemplate"] = foundTemplate

        return foundEntities

    def cleanEntitiesDict(self, entitiesDict, replacementList=[]):
        cleanedEntitiesDict = {}
        replacementList = ["invoice", "date"]
        if replacementList:
            replacementList = replacementList

        for entity in entitiesDict:
            subList = entitiesDict[entity]
            cleanedSublist = "".join(subList)
            for word in replacementList:
                foundIndex = cleanedSublist.lower().find(word)
                if foundIndex != -1:
                    cleanedSublist = cleanedSublist[:foundIndex] + cleanedSublist[(len(word) + foundIndex):]
            cleanedEntitiesDict[entity] = cleanedSublist

        return cleanedEntitiesDict

    def getGoldLabels(self, dataInstance):
        goldLabels = dataInstance["goldLabels"]
        return {k: v["value"] for k, v in goldLabels.items() if v is not None}

    def testTemplateApproach(self, dataset):

        testResults = {}
        for i in range(len(dataset)):
            dataInstance = dataset[i]
            name = dataInstance["instanceFolderPath"].split("\\")[-1]
            print(i, name)
            testResults[name] = {}
            testResults[name]["predictions"] = self.findEntities(dataInstance)
            testResults[name]["goldLabels"] = self.getGoldLabels(dataInstance)

        time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        createJSON(f"F:\\CodeCopy\\InvoiceInformationExtraction\\Intellix_based\\testResults_{time}.json", testResults)


