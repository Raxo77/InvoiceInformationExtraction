import math
import time
import string
import runnerAll
import Levenshtein
import numpy as np
import pandas as pd
from datetime import datetime
from dataProcessing.customDataset import CustomDataset
from utils.helperFunctions import loadJSON, getConfig, CONFIG_PATH, flattenList, createJSON

def getSequence(dataInstance) -> str:
    # As opposed to BERT-features.csv (i.e., with punct) this .csv contains no entries consisting only
    # of punctuation characters <--> allows to streamline the process while still retaining essential/contextual
    # information conveyed by punctuation character
    # e.g. for phone numbers separated with "-" - w/o this separation might also be an IBAN
    featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
    colNames = list(featuresDF.columns)
    colNames[0] = "wordKey"
    featuresDF.columns = colNames

    seqString = ""
    seqList = list(map(lambda x: x.split("_")[0], featuresDF["wordKey"]))

    seqString += seqList[0]
    punct = string.punctuation

    for i in seqList[1:]:
        if i in punct:
            seqString += i
        else:
            seqString += f" {i}"

    return seqString


def findLongestCommonSubstring(strings):
    if not strings:
        return ""

    # Generate all possible substrings of the first string
    def allSubstrings(s):
        substrings = set()
        length = len(s)
        for i in range(length):
            for j in range(i + 1, length + 1):
                substrings.add(s[i:j])
        return substrings

    # Find common substrings in all strings
    def commonSubstrings(substrings, strings):
        common = set(substrings)
        for s in strings:
            current_substrings = allSubstrings(s)
            common = common.intersection(current_substrings)
            if not common:
                return set()
        return common

    # Get all substrings of the first string
    first_string_substrings = allSubstrings(strings[0])

    # Find common substrings in all strings
    common_substrings = commonSubstrings(first_string_substrings, strings[1:])

    # If there are no common substrings, return an empty string
    if not common_substrings:
        return ""

    # Find the longest common substring
    longest_common_substring = ""
    for substring in common_substrings:
        if len(substring) > len(longest_common_substring):
            longest_common_substring = substring

    return longest_common_substring


class InvoiceEnsemble:

    def __init__(self,
                 employmentDetailsList: list[bool]
                 ):

        self.labelTranslationStandard = {'invoiceDate': 1,
                                         'invoiceNumber': 2,
                                         'invoiceGrossAmount': 3,
                                         'invoiceTaxAmount': 4,
                                         'orderNumber': 5,
                                         'issuerName': 6,
                                         'issuerIBAN': 7,
                                         'issuerAddress': 8,
                                         "issuerCity": 9
                                         }

        self.labelTranslationIOB = {'B-invoiceDate': 1,
                                    'I-invoiceDate': 2,
                                    'B-invoiceNumber': 3,
                                    'I-invoiceNumber': 4,
                                    'B-invoiceGrossAmount': 5,
                                    'I-invoiceGrossAmount': 6,
                                    'B-invoiceTaxAmount': 7,
                                    'I-invoiceTaxAmount': 8,
                                    'B-orderNumber': 9,
                                    'I-orderNumber': 10,
                                    'B-issuerName': 11,
                                    'I-issuerName': 12,
                                    'B-issuerIBAN': 13,
                                    'I-issuerIBAN': 14,
                                    'B-issuerAddress': 15,
                                    'I-issuerAddress': 16,
                                    'B-issuerCity': 17,
                                    'I-issuerCity': 18,
                                    'O': 0
                                    }

        self.employmentInfo = employmentDetailsList
        self.mappedPredictions = {}
        self.finalPredictions = {}
        self.testData = CustomDataset(getConfig("pathToTestDataFolder", CONFIG_PATH))
        #self.testData = CustomDataset(getConfig("pathToTestDataFolderNewTemplatesOnly", CONFIG_PATH))

    def employSubmodel1(self):

        if not self.employmentInfo[0]:
            print("Submodel 1 - BERT-CRF NOT employed for the ensemble")
            return None

        # testResultsSM1 = runnerAll.subModel1(train=False, test=True)
        #testResultsSM1 = loadJSON(
        #    r"F:\CodeCopy\InvoiceInformationExtraction\BERT_based\testResults_11-06-2024_10-14-34.json")
        testResultsSM1 = loadJSON(
            r"F:\CodeCopy\InvoiceInformationExtraction\BERT_based\testResults_10-06-2024_21-00-49.json")

        mappedPredictions = {}
        self.finalPredictions["BERT-CRF_based"] = {}

        for itemNum in testResultsSM1.keys():

            mappedPredictions[itemNum] = {}
            mappedPredictions[itemNum]["predictions"] = []
            mappedPredictions[itemNum]["goldLabels"] = []
            mappedPredictions[itemNum]["jointWords"] = []

            instancePredictions = testResultsSM1[itemNum]["predictions"]
            instanceLabels = testResultsSM1[itemNum]["goldLabels"]

            tokensPerWord = []
            currentIdx = 0
            for count, token in enumerate(testResultsSM1[itemNum]["instanceTokens"][::-1]):
                if len(tokensPerWord) <= currentIdx:
                    tokensPerWord.append(0)
                if token.startswith("##"):
                    tokensPerWord[currentIdx] += 1

                else:
                    tokensPerWord[currentIdx] += 1
                    currentIdx += 1
            tokensPerWord = tokensPerWord[::-1]

            previousWordIdx = 0
            for tokenCount in tokensPerWord:
                subListPredictions = instancePredictions[previousWordIdx:tokenCount + previousWordIdx]
                subListGoldLabels = instanceLabels[previousWordIdx:tokenCount + previousWordIdx]

                mappedPredictions[itemNum]["predictions"].append(
                    max(set(subListPredictions), key=subListPredictions.count))

                mappedPredictions[itemNum]["jointWords"].append("".join(
                    testResultsSM1[itemNum]["instanceTokens"][previousWordIdx:tokenCount + previousWordIdx]).replace(
                    "#", ""))

                mappedPredictions[itemNum]["goldLabels"].append(
                    max(set(subListGoldLabels), key=subListGoldLabels.count))

                previousWordIdx += tokenCount

            self.finalPredictions["BERT-CRF_based"][itemNum] = {k: [] for k in self.labelTranslationStandard.values()}

            for labelNumber in self.finalPredictions["BERT-CRF_based"][itemNum].keys():
                values = np.array(mappedPredictions[itemNum]["predictions"])

                for foundIndex in np.where(values == labelNumber)[0].tolist():
                    self.finalPredictions["BERT-CRF_based"][itemNum][labelNumber].append(
                        mappedPredictions[itemNum]["jointWords"][foundIndex])

                self.finalPredictions["BERT-CRF_based"][itemNum][labelNumber] = "".join(
                    self.finalPredictions["BERT-CRF_based"][itemNum][labelNumber])

            self.finalPredictions["BERT-CRF_based"][itemNum] = {
                k: self.finalPredictions["BERT-CRF_based"][itemNum][v] for k, v in
                self.labelTranslationStandard.items()}

        self.mappedPredictions["BERT-CRF_based"] = mappedPredictions

    def employSubmodel2(self):

        if not self.employmentInfo[1]:
            print("Submodel 2 - CloudScan-based approach NOT employed for the ensemble")
            return None

        self.finalPredictions["CloudScan_based"] = {}

        # testResultsSM2 = runnerAll.subModel2(train=False, test=True)
        testResultsSM2 = loadJSON(
            r"F:\CodeCopy\InvoiceInformationExtraction\CloudScan_based\testResults_10-06-2024_21-11-13.json")
        #testResultsSM2 = loadJSON(
         #   r"F:\CodeCopy\InvoiceInformationExtraction\CloudScan_based\testResults_11-06-2024_10-18-26.json")

        for itemNum in testResultsSM2:

            predictions = testResultsSM2[itemNum]["predictions"]
            words = testResultsSM2[itemNum]["instanceTokens"]

            self.finalPredictions["CloudScan_based"][itemNum] = {k: [] for k in self.labelTranslationIOB.values()}
            for labelNum in self.labelTranslationIOB.values():
                values = np.array(predictions)

                for foundIndex in np.where(values == labelNum)[0].tolist():
                    self.finalPredictions["CloudScan_based"][itemNum][labelNum].append(
                        words[foundIndex]
                    )
                self.finalPredictions["CloudScan_based"][itemNum][labelNum] = "".join(
                    self.finalPredictions["CloudScan_based"][itemNum][labelNum])

            tempDict = {k: "" for k in self.labelTranslationStandard.values()}
            for labelIdx in self.finalPredictions["CloudScan_based"][itemNum].keys():
                if labelIdx == 0:
                    continue

                tempDict[math.ceil(labelIdx / 2)] += self.finalPredictions["CloudScan_based"][itemNum][labelIdx]
            tempDict = {k: tempDict[v] for k, v in self.labelTranslationStandard.items()}
            self.finalPredictions["CloudScan_based"][itemNum] = tempDict

    def employSubmodel3(self):

        if not self.employmentInfo[2]:
            print("Submodel 3 - BBMC-based approach NOT employed for the ensemble")
            return None

        # testResultsSM3 = runnerAll.subModel3(train=False, test=True)
        #testResultsSM3 = loadJSON(
        #    r"F:\CodeCopy\InvoiceInformationExtraction\BBMC_based\testResults11-06-2024_10-21-43.json")
        testResultsSM3 = loadJSON(
            r"F:\CodeCopy\InvoiceInformationExtraction\BBMC_based\testResults10-06-2024_21-17-49.json")

        mappedPredictions = {}
        self.finalPredictions["BBMC_based"] = {}

        for itemNum in testResultsSM3.keys():

            mappedPredictions[itemNum] = {}
            mappedPredictions[itemNum]["predictions"] = []
            mappedPredictions[itemNum]["goldLabels"] = []
            mappedPredictions[itemNum]["jointWords"] = []

            instancePredictions = testResultsSM3[itemNum]["predictions"]
            instanceLabels = testResultsSM3[itemNum]["goldLabels"]

            tokensPerWord = []
            currentIdx = 0
            for count, token in enumerate(testResultsSM3[itemNum]["instanceTokens"][::-1]):
                if len(tokensPerWord) <= currentIdx:
                    tokensPerWord.append(0)
                if token.startswith("##"):
                    tokensPerWord[currentIdx] += 1

                else:
                    tokensPerWord[currentIdx] += 1
                    currentIdx += 1
            tokensPerWord = tokensPerWord[::-1]

            previousWordIdx = 0
            for tokenCount in tokensPerWord:
                subListPredictions = instancePredictions[previousWordIdx:tokenCount + previousWordIdx]
                subListGoldLabels = instanceLabels[previousWordIdx:tokenCount + previousWordIdx]

                mappedPredictions[itemNum]["predictions"].append(
                    max(set(subListPredictions), key=subListPredictions.count))

                mappedPredictions[itemNum]["jointWords"].append("".join(
                    testResultsSM3[itemNum]["instanceTokens"][previousWordIdx:tokenCount + previousWordIdx]).replace(
                    "#", ""))

                mappedPredictions[itemNum]["goldLabels"].append(
                    max(set(subListGoldLabels), key=subListGoldLabels.count))

                previousWordIdx += tokenCount

            self.finalPredictions["BBMC_based"][itemNum] = {k: [] for k in self.labelTranslationIOB.values()}
            for labelNum in self.labelTranslationIOB.values():
                values = np.array(mappedPredictions[itemNum]["predictions"])

                for foundIndex in np.where(values == labelNum)[0].tolist():
                    self.finalPredictions["BBMC_based"][itemNum][labelNum].append(
                        mappedPredictions[itemNum]["jointWords"][foundIndex]
                    )

                self.finalPredictions["BBMC_based"][itemNum][labelNum] = "".join(
                    self.finalPredictions["BBMC_based"][itemNum][labelNum])

            tempDict = {k: "" for k in self.labelTranslationStandard.values()}
            for labelIdx in self.finalPredictions["BBMC_based"][itemNum].keys():
                if labelIdx == 0:
                    continue

                tempDict[math.ceil(labelIdx / 2)] += self.finalPredictions["BBMC_based"][itemNum][labelIdx]
            tempDict = {k: tempDict[v] for k, v in self.labelTranslationStandard.items()}
            self.finalPredictions["BBMC_based"][itemNum] = tempDict

    def employSubmodel4(self):

        if not self.employmentInfo[3]:
            print("Submodel 4 - BiLSTM_CNN_CRF-based approach NOT employed for the ensemble")
            return None

        # testResultsSM4 = runnerAll.subModel4(train=False, test=True)
        #testResultsSM4 = loadJSON(
        #    r"F:\CodeCopy\InvoiceInformationExtraction\BiLSTM_CNN_CRF_based\testResults11-06-2024_10-58-21.json")
        testResultsSM4 = loadJSON(
            r"F:\CodeCopy\InvoiceInformationExtraction\BiLSTM_CNN_CRF_based\testResults10-06-2024_21-46-01.json")

        self.finalPredictions["BiLSTM_CNN_CRF_based"] = {}
        for count, itemNum in enumerate(testResultsSM4):

            dataInstance = self.testData[count]
            words = getSequence(dataInstance).split(" ")

            self.finalPredictions["BiLSTM_CNN_CRF_based"][itemNum] = {k: [] for k in self.labelTranslationIOB.values()}
            for labelNum in self.labelTranslationIOB.values():
                values = np.array(testResultsSM4[itemNum]["predictions"])

                for foundIndex in np.where(values == labelNum)[0].tolist():
                    self.finalPredictions["BiLSTM_CNN_CRF_based"][itemNum][labelNum].append(
                        words[foundIndex])

                self.finalPredictions["BiLSTM_CNN_CRF_based"][itemNum][labelNum] = "".join(
                    self.finalPredictions["BiLSTM_CNN_CRF_based"][itemNum][labelNum])

            tempDict = {k: "" for k in self.labelTranslationStandard.values()}
            for labelIdx in self.finalPredictions["BiLSTM_CNN_CRF_based"][itemNum].keys():
                if labelIdx == 0:
                    continue

                tempDict[math.ceil(labelIdx / 2)] += self.finalPredictions["BiLSTM_CNN_CRF_based"][itemNum][labelIdx]
            tempDict = {k: tempDict[v] for k, v in self.labelTranslationStandard.items()}
            self.finalPredictions["BiLSTM_CNN_CRF_based"][itemNum] = tempDict

    def employSubmodel5(self):

        if not self.employmentInfo[4]:
            print("Submodel 5 - GCN-based approach NOT employed for the ensemble")
            return None

        # testResultsSM5 = runnerAll.subModel5(train=False, test=True)
        #testResultsSM5 = loadJSON(
        #    r"F:\CodeCopy\InvoiceInformationExtraction\GCN_based\testResults_11-06-2024_10-59-35.json")
        testResultsSM5 = loadJSON(
            r"F:\CodeCopy\InvoiceInformationExtraction\GCN_based\testResults_10-06-2024_21-49-37.json")

        self.finalPredictions["GCN_based"] = {}
        for count, itemNum in enumerate(testResultsSM5):
            self.finalPredictions["GCN_based"][itemNum] = {k: [] for k in self.labelTranslationIOB.values()}
            words = testResultsSM5[itemNum]["instanceTokens"]

            for labelNum in self.labelTranslationIOB.values():
                values = np.array(testResultsSM5[itemNum]["predictions"])

                for foundIndex in np.where(values == labelNum)[0].tolist():
                    self.finalPredictions["GCN_based"][itemNum][labelNum].append(
                        words[foundIndex])

                self.finalPredictions["GCN_based"][itemNum][labelNum] = "".join(
                    self.finalPredictions["GCN_based"][itemNum][labelNum])

            tempDict = {k: "" for k in self.labelTranslationStandard.values()}
            for labelIdx in self.finalPredictions["GCN_based"][itemNum].keys():
                if labelIdx == 0:
                    continue

                tempDict[math.ceil(labelIdx / 2)] += self.finalPredictions["GCN_based"][itemNum][labelIdx]
            tempDict = {k: tempDict[v] for k, v in self.labelTranslationStandard.items()}
            self.finalPredictions["GCN_based"][itemNum] = tempDict

    def employSubmodel6(self):



        testResults = loadJSON(
            r"F:\CodeCopy\InvoiceInformationExtraction\Intellix_based\testResults_10-06-2024_22-11-36.json")

        testResultsOnly = {k: testResults[k]["predictions"] for k in testResults}
        goldLabelsOnly = {k: testResults[k]["goldLabels"] for k in testResults}
        self.finalPredictions["goldLabelsOnly"] = goldLabelsOnly
        if not self.employmentInfo[5]:
            print("Submodel 6 - Intellix-based approach NOT employed for the ensemble")
            return None
        self.finalPredictions["Intellix_based"] = testResultsOnly

    def assessResults(self):

        self.employSubmodel1()
        self.employSubmodel2()
        self.employSubmodel3()
        self.employSubmodel4()
        self.employSubmodel5()
        self.employSubmodel6()

        temp = {}
        models = [i for i in self.finalPredictions.keys() if i != "goldLabelsOnly"]
        entityList = [i for i in getConfig("targetLabels", CONFIG_PATH).keys()]

        for sampleNumber in self.finalPredictions[models[0]].keys():
            temp[sampleNumber] = {}
            for entity in entityList:
                temp[sampleNumber][entity] = []
                for model in models:
                    try:
                        prediction = self.finalPredictions[model][sampleNumber][entity]
                    except KeyError:
                        continue
                    if not prediction:
                        continue
                    temp[sampleNumber][entity].append(prediction)


        ensemblePredictions = {}
        for sampleNumber, subDict in temp.items():
            ensemblePredictions[sampleNumber] = {}
            for entity in subDict.keys():

                focalElement = temp[sampleNumber][entity]
                if len(focalElement) == 0:
                    currentPrediction = ""

                elif len(set(focalElement)) == 1:
                    currentPrediction = temp[sampleNumber][entity][0]

                else:
                    currentPrediction = findLongestCommonSubstring(focalElement)
                ensemblePredictions[sampleNumber][entity] = currentPrediction

        self.finalPredictions["ensemblePredictions"] = ensemblePredictions

        for model in [i for i in self.finalPredictions.keys() if i != "goldLabelsOnly"]:
            for sampleNum, subDict in self.finalPredictions[model].items():
                deltaPerSample = 0
                compLen1 = 0
                compLen2 = 0
                for entity, proposal in subDict.items():
                    if entity == "foundTemplate":
                        continue
                    compLen1 += len(proposal)
                    try:
                        groundTruth = self.finalPredictions["goldLabelsOnly"][sampleNum][entity].replace(" ", "")
                        compLen2 += len(groundTruth)
                    except KeyError:
                        continue
                    deltaPerSample += Levenshtein.distance(proposal, groundTruth)
                self.finalPredictions[model][sampleNum]["Levenshtein"] = deltaPerSample
                self.finalPredictions[model][sampleNum]["Similarity"] = 1 - (
                        deltaPerSample / max(compLen1, compLen2)) if max(compLen1, compLen2) != 0 else None

        time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        createJSON(f"./finalPredictionsAndResults_{time}.json", self.finalPredictions)
        print("")

