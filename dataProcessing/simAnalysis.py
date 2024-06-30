import torch
import shutil
import statistics
import pandas as pd
from customDataset import CustomDataset
from filterRawDataset import listDirectory
from utils.helperFunctions import loadJSON

torch.manual_seed(123)


def getSimAnaResults():
    # df = pd.DataFrame(data={"itemNumber": [], "similarity": [], "templateNumber": []})
    df = pd.read_csv("./simAnalysisResults.csv", index_col=0)

    data = CustomDataset(r"F:\testData")
    for i in range(len(data)):
        print(i)

        dataInstance = data[i]

        itemNum = dataInstance["instanceFolderPath"].split("\\")[-1]

        pathToSim = dataInstance["instanceFolderPath"] + "\\hOCR_groundWords_similarity.csv"
        sim = pd.read_csv(pathToSim).loc[0, "similarity"]

        pathToTemplateNum = dataInstance["instanceFolderPath"] + "\\details.json"
        templateNum = loadJSON(pathToTemplateNum)["invoice"]["template"].split("/")[-1][8:-4]

        df.loc[len(df)] = [itemNum, sim, templateNum]

    df.to_csv("./simAnalysisResults.csv")


def createSampledSubset(lenSubset=5000):
    data = CustomDataset(r"F:\trainDataAll")
    indexPerm = torch.randperm(len(data))
    trainSubsetIndices = indexPerm[2000:2000 + lenSubset]

    trainTemplates = pd.unique(pd.read_csv("samplesTrainSubset.csv").TemplateNumber)

    filesAdded = 0
    i = 0
    while filesAdded < lenSubset:
        dataInstance = data[i]
        i += 1

        sourcePath = dataInstance["instanceFolderPath"]

        if loadJSON(f"{sourcePath}\\details.json")["invoice"]["template"].split("/")[-1][:-4] not in trainTemplates:
            destinationPath = "F:/testDataSubsetNewTemplatesOnly/" + sourcePath.split("\\")[-1]

            shutil.copytree(sourcePath, destinationPath)
            filesAdded += 1


def indexBasedResultComparison(pathToJSON):
    data = loadJSON(pathToJSON)
    indexMatchList = {}
    for key, resultDict in data.items():
        goldLabels = resultDict["goldLabels"]
        predictions = resultDict["predictions"]

        nonzero_indices_list1 = pd.DataFrame([index for index, value in enumerate(goldLabels) if value != 0])
        nonzero_indices_list2 = pd.DataFrame([index for index, value in enumerate(predictions) if value != 0])

        if not nonzero_indices_list1.empty and not nonzero_indices_list2.empty:
            indexMatchList[key] = len(
                pd.merge(nonzero_indices_list1, nonzero_indices_list2, how="inner").iloc[:, 0].tolist()) / len(
                pd.merge(nonzero_indices_list1, nonzero_indices_list2, how="outer").iloc[:, 0].tolist())
        else:
            # in certain cases (e.g. invoice 10080) no gold labels are provided
            # OR - and this occurs quite often with e.g. BERT-CRF model:
            # only 0 is predicted
            indexMatchList[key] = 0.

    return indexMatchList


def getCornerstoneInfo(trainPath,
                       testPath
                       ):
    samplesTrainSubset = [[i.name,
                           loadJSON(f"{i.path}\\details.json")["invoice"]["template"].split("/")[-1][:-4],
                           round(pd.read_csv(f"{i.path}\\hOCR_groundWords_similarity.csv").similarity.item(), 2)]
                          for i
                          in listDirectory(trainPath)]

    samplesTestSubset = [[i.name,
                          loadJSON(f"{i.path}\\details.json")["invoice"]["template"].split("/")[-1][:-4],
                          round(pd.read_csv(f"{i.path}\\hOCR_groundWords_similarity.csv").similarity.item(), 2)]
                         for i in listDirectory(testPath)]

    samplesInBoth1 = [i for i in samplesTestSubset if i in samplesTrainSubset]

    noOverlappingSamples = False
    if len(samplesInBoth1) == 0:
        noOverlappingSamples = True
    print(noOverlappingSamples)

    samplesTestSubset = pd.DataFrame(samplesTestSubset, columns=["SampleNumber", "TemplateNumber", "Similarity"])
    samplesTrainSubset = pd.DataFrame(samplesTrainSubset, columns=["SampleNumber", "TemplateNumber", "Similarity"])

    samplesTrainSubset.to_csv("samplesTrainAll.csv")
    samplesTestSubset.to_csv("samplesTestAll.csv")

    return noOverlappingSamples

