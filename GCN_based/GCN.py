import torch
import string
import os.path
import numpy as np
import pandas as pd
import networkx as nx
import torch_geometric.data
from datetime import datetime
import matplotlib.pyplot as plt
from torch_geometric.nn.conv import ChebConv
from torch_geometric.loader import DataLoader
from transformers import BertTokenizerFast, BertModel
from dataProcessing.customDataset import CustomDataset
from GCN_based.featureExtraction import featureCalculation
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.helperFunctions import getConfig, CONFIG_PATH, createJSON, loadJSON

torch.manual_seed(123)

TOKENIZER = BertTokenizerFast.from_pretrained('bert-base-cased')
MODEL = BertModel.from_pretrained('bert-base-cased')
tokenizer = TOKENIZER
model = MODEL


def countTokensPerWord(text, offset_mapping):
    # Split the text into words by whitespace
    words = text.split()
    word_boundaries = []
    index = 0

    # Calculate the start and end indices for each word
    for word in words:
        start_index = text.index(word, index)
        end_index = start_index + len(word)
        word_boundaries.append((start_index, end_index))
        index = end_index

    # Initialize the list for counting tokens per word
    token_counts = [0] * len(words)

    # Assign tokens to words based on offset mappings
    for token_start, token_end in offset_mapping:
        for i, (word_start, word_end) in enumerate(word_boundaries):
            # Check if the token falls within the boundaries of the word
            if word_start <= token_start < word_end:
                token_counts[i] += 1
                break

    return token_counts


class InvoiceGCN(torch.nn.Module):

    def __init__(self,
                 numClasses=getConfig("GCN_based", CONFIG_PATH)["numLabels"],
                 model=model,
                 tokenizer=tokenizer,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 initFilterNumber=16,
                 filterSize=3,
                 featureSize=782,
                 batchSize=64
                 ):
        super(InvoiceGCN, self).__init__()

        self.initFilterNumber = initFilterNumber
        self.numClasses = numClasses

        self.device = device
        self.batchSize = batchSize
        self.filterSize = filterSize

        self.tokenizer = tokenizer
        self.embeddings = model.embeddings.to(self.device)

        self.relu = torch.nn.ReLU()
        self.gcn1 = ChebConv(featureSize, self.initFilterNumber, K=self.filterSize).to(self.device)
        self.gcn2 = ChebConv(self.initFilterNumber, 2 * self.initFilterNumber, K=self.filterSize).to(self.device)
        self.gcn3 = ChebConv(2 * self.initFilterNumber, 4 * self.initFilterNumber, K=self.filterSize).to(self.device)
        self.gcn4 = ChebConv(4 * self.initFilterNumber, 8 * self.initFilterNumber, K=self.filterSize).to(self.device)
        self.gcn5 = ChebConv(8 * self.initFilterNumber, self.numClasses, K=self.filterSize).to(self.device)
        self.softmax = torch.nn.Softmax()

    def getSequence(self, dataInstance):

        featuresDF = pd.read_csv(dataInstance["BERT-basedNoPunctFeaturesPath"])
        return " ".join(list(map(lambda x: str(x).split("_")[0], featuresDF["wordKey"])))

    def embedWords(self, wordSeq):

        tokens = self.tokenizer(wordSeq, return_offsets_mapping=True)
        offsets = tokens.encodings[0].offsets

        tokensPerWord = countTokensPerWord(wordSeq, offsets)

        tokenEmbeddings = self.embeddings(torch.tensor([tokens["input_ids"]]).to(self.device))

        wordEmbeds = []
        wordIndex = 0
        index = 0
        for i in tokensPerWord:
            wordEmbeds.append(tokenEmbeddings[0, wordIndex:wordIndex + i, :].mean(dim=0))
            index += i

        wordEmbeds = torch.stack(wordEmbeds)
        return wordEmbeds

    def plotGraph(self,
                  data,
                  showGraph=False,
                  saveGraph=False,
                  instanceFolderPath=""
                  ):
        G = nx.Graph()
        edge_index = data.edge_index.cpu().numpy()
        pos = {i: (coords[0], abs(2200 - coords[1])) for i, coords in enumerate(data.pos.cpu().numpy())}
        edges = zip(edge_index[0], edge_index[1])
        G.add_edges_from(edges)
        wordLabels = {i: data.words[i] for i in range(len(data.words))}

        nx.draw(G, pos, with_labels=False, node_size=70, node_color="lightblue", alpha=1, edge_color="gray")
        nx.draw_networkx_labels(G, pos, labels=wordLabels, font_size=9)
        if saveGraph:
            plt.savefig(os.path.join(instanceFolderPath, "invoiceGraph.png"))
        if showGraph:
            plt.show()

    def getNodes(self, dataInstance):
        invoiceSeq = self.getSequence(dataInstance)

        nodeWords = invoiceSeq.split()
        wordEmbeddings = self.embedWords(invoiceSeq)
        customFeatures = featureCalculation(invoiceSeq, dataInstance)
        words, temp = [], []
        for subList in customFeatures:
            words.append(subList[1])
            temp.append(subList[-1])
        customFeatures = temp
        customFeatures = torch.tensor(customFeatures).to(self.device)
        nodeFeatures = torch.cat((wordEmbeddings, customFeatures), dim=1)
        nodeLabels, _, __ = self.getGoldLabels(dataInstance)
        nodeLabels = torch.tensor(nodeLabels, dtype=torch.long)

        return nodeWords, nodeFeatures, nodeLabels

    def getGoldLabels(self,
                      dataInstance,
                      feedback=False
                      ):

        featuresDF = pd.read_csv(dataInstance["BERT-basedNoPunctFeaturesPath"])

        goldLabelsChar = ["O" for i in range(len(featuresDF))]
        groundTruth = dataInstance["goldLabels"]
        hOCRcharSeq = "".join(list(map(lambda x: x.split("_")[0], featuresDF["wordKey"])))
        groundTruthCharSeq = [i["value"].replace(" ", "") for i in groundTruth.values() if i is not None]
        b = (list(map(lambda x: x.split("_")[0], featuresDF["wordKey"])))
        labelTranslation = {f"{tag}-{i}": (2 * count + 1) + (counter * 1) for count, i in
                            enumerate(groundTruth.keys()) for
                            counter, tag in enumerate(["B", "I"])}
        labelTranslation["O"] = 0

        for label, i in zip([i for i in groundTruth.keys() if groundTruth[i] is not None], groundTruthCharSeq):
            a = hOCRcharSeq.find(i)

            tempLen = 0
            targetLen = a + len(i)
            firstFind = True
            for index, j in enumerate(b):
                if a <= tempLen < targetLen:
                    if firstFind:
                        goldLabelsChar[index] = f"B-{label}"
                    else:
                        goldLabelsChar[index] = f"I-{label}"
                    firstFind = False

                tempLen += len(j)

        if feedback:
            for i, j in zip(goldLabelsChar, b):
                print(i, j)

        goldLabels = [labelTranslation[i] for i in goldLabelsChar]
        return goldLabels, goldLabelsChar, labelTranslation

        # For each gold label, match  char sequence of hOCR text with char sequence of groundTruth
        # This is done to circumnavigate issues arising from different "tokenization" of hOCR and groundTruth strings

    def getEdges(self,
                 dataInstance
                 ):

        featuresDF = pd.read_csv(dataInstance["BERT-basedNoPunctFeaturesPath"], dtype={"wordKey": "str"})

        edgeList = []
        nodeNameToIndex = {}

        count = 0
        for i in featuresDF.loc[:, "wordKey"].tolist():
            if i.split("_")[0] not in string.punctuation:
                nodeNameToIndex[i] = count
                count += 1

        for word in featuresDF.loc[:, "wordKey"].tolist():

            for direction in ["left", "right", "above", "below"]:
                temp = featuresDF.loc[featuresDF["wordKey"] == word, direction].item()
                if not pd.isna(temp):
                    edgeList.append([nodeNameToIndex[word],
                                     nodeNameToIndex[temp]])

        edgeTensor = torch.tensor(edgeList, dtype=torch.long)
        return edgeTensor

    def graphModeller(self, dataInstance):
        nodeWords, nodeFeatures, nodeLabels = self.getNodes(dataInstance)
        edgeIndex = self.getEdges(dataInstance)

        featuresDF = pd.read_csv(dataInstance["BERT-basedNoPunctFeaturesPath"])

        coords = [[int(i.split("_")[-2]), int(i.split("_")[-1])] for i in featuresDF.loc[:, "wordKey"].tolist()]
        coords = torch.tensor(coords)

        data = torch_geometric.data.Data(x=nodeFeatures,
                                         edge_index=edgeIndex.t().contiguous(),
                                         y=nodeLabels)
        data.words = nodeWords
        data.pos = coords

        return data

    def forward(self, graphData) -> torch.Tensor:

        nodes = graphData.x
        edgeIndex = graphData.edge_index
        batch = graphData.batch

        x = self.relu(self.gcn1(nodes, edgeIndex))
        x = self.relu(self.gcn2(x, edgeIndex))
        x = self.relu(self.gcn3(x, edgeIndex))
        x = self.relu(self.gcn4(x, edgeIndex))
        x = self.gcn5(x, edgeIndex)
        x = torch.nn.functional.softmax(x)

        return x

    def trainModel(self,
                   numEpochs,
                   dataset,
                   trainHistoryPath="",
                   lr=1e-4,
                   saveInvoiceGraph=False,
                   ):

        if trainHistoryPath:
            trainHistory = pd.read_csv(trainHistoryPath)

        try:
            epochData = pd.read_csv("./trainEpochData_06-06.csv")
        except FileNotFoundError:
            epochData = pd.DataFrame(columns=['epoch', 'avgLoss'])

        try:
            batchData = loadJSON(f"./batchData.json")
        except FileNotFoundError:
            batchData = {}

        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)
        criterion = torch.nn.CrossEntropyLoss()

        # outermost loop - handles number of epochs
        for epoch in range(numEpochs):
            print(f"Epoch {epoch + 1} / {numEpochs}")

            overallEpochLoss = 0
            shuffledIndices = torch.randperm(len(dataset))

            batchSize = self.batchSize

            # intermediate loop - handles batches
            for i in range(batchSize, len(dataset), batchSize):

                # Notably, with this approach the last batch per epoch is omitted
                # --> as long as batch size is small in relation to len(dataset), this is inconsequential,
                # with considerable batch sizes though, this needs to be considered/adjusted

                batchDataIndex = f"{epoch}_{i}"
                batchData[batchDataIndex] = {"batchLoss": 0,
                                             "batchItems": [],
                                             "goldLabels": [],
                                             "predictions": []
                                             }

                allInstances = shuffledIndices[i - batchSize:i]
                graphDataList = []
                nodesList = []
                edgeIndicesList = []
                labelsList = []

                # innermost loop - respectively handles concrete instances in each batch
                for batchNum, idx in enumerate(allInstances):

                    dataInstance = dataset[idx]

                    pathToInstance = dataInstance["instanceFolderPath"]
                    batchData[batchDataIndex]["batchItems"].append(pathToInstance.split("\\")[-1])

                    print(i, batchNum, pathToInstance)
                    itemNum = pathToInstance.split("\\")[-1]

                    if trainHistoryPath and f"{itemNum}_{epoch}" in trainHistory.values:
                        continue

                    if trainHistoryPath:
                        trainHistory.loc[len(trainHistory)] = f"{itemNum}_{epoch}"

                    graphData = self.graphModeller(dataInstance)
                    if saveInvoiceGraph:
                        self.plotGraph(graphData, saveGraph=saveInvoiceGraph, instanceFolderPath=pathToInstance)
                    graphDataList.append(graphData)
                    goldLabels = graphData.y

                    labelsList.append(goldLabels)
                    nodesList.append(graphData.x)
                    edgeIndicesList.append(graphData.edge_index)

                if not labelsList:
                    continue

                goldLabels = torch.cat(labelsList, dim=0).to(self.device)

                batchData[batchDataIndex]["goldLabels"].append(goldLabels.tolist())

                self.zero_grad()

                dataLoader = DataLoader(graphDataList, batchSize, shuffle=False)
                graphData = next(iter(dataLoader)).to(self.device)

                x = self.forward(graphData)

                predictions = torch.argmax(x, dim=1)

                batchData[batchDataIndex]["predictions"].append(predictions.tolist())

                loss = criterion(x, goldLabels)
                overallEpochLoss += loss.item()
                batchData[batchDataIndex]["batchLoss"] = loss.item()

                loss.backward()
                optimizer.step()

            createJSON(r"F:\CodeCopy\InvoiceInformationExtraction\GCN_based\batchData.json", batchData)

            if trainHistoryPath:
                trainHistory.to_csv(trainHistoryPath, index=False)

            time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            checkpointPath = getConfig("GCN_based", CONFIG_PATH)["pathToStateDict"][:-3] + f"{epoch}_"
            checkpointPath += time
            checkpointPath += ".pt"
            torch.save(self.state_dict(), checkpointPath)

            overallEpochLoss = overallEpochLoss / ((len(dataset) // self.batchSize) * self.batchSize)
            epochData = epochData.append({'epoch': epoch + 1, 'avgLoss': overallEpochLoss}, ignore_index=True)
            scheduler.step()

            epochData.to_csv(r"F:\CodeCopy\InvoiceInformationExtraction\GCN_based\trainEpochData.csv", index=False)

        torch.save(self.state_dict(), getConfig("BiLSTM_CNN_CRF_based", CONFIG_PATH)["pathToStateDict"])
        print("Training of GCN-based model complete")

    def testModel(self,
                  dataset
                  ):

        criterion = torch.nn.CrossEntropyLoss()
        testResults = {}
        self.eval()

        with torch.no_grad():
            overallAcc = 0
            overallLoss = 0

            for c, idx in enumerate(range(len(dataset))):
                dataInstance = dataset[idx]
                pathToInstance = dataInstance["instanceFolderPath"]

                print(c, pathToInstance)

                itemNum = pathToInstance.split("\\")[-1]
                testResults[itemNum] = {}

                graphData = self.graphModeller(dataInstance).to(self.device)
                goldLabels = graphData.y

                x = self.forward(graphData)

                loss = criterion(x, goldLabels)
                overallLoss += loss.item()

                predictions = torch.argmax(x, dim=1)

                testResults[itemNum]["goldLabels"] = goldLabels.tolist()

                testResults[itemNum]["instanceTokens"] = graphData.words
                testResults[itemNum]["predictions"] = predictions.tolist()
                testResults[itemNum]["loss"] = loss.item()
                testResults[itemNum]["NumberOfNonZeroLabelsPredicted"] = sum(
                    list(map(lambda x: 1 if x != 0 else 0, predictions.tolist())))
                testResults[itemNum]["NumberOfNonZeroLabelsGold"] = sum(
                    list(map(lambda x: 1 if x != 0 else 0, goldLabels.tolist())))

                testResults[itemNum]["accuracy"] = accuracy_score(goldLabels.tolist(),
                                                                  predictions.tolist())

                overallAcc += testResults[itemNum]["accuracy"]

                testResults[itemNum]["confusionMatrix"] = confusion_matrix(
                    goldLabels.tolist(),
                    predictions.tolist()).tolist()

            time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            createJSON(f"F:\\CodeCopy\\InvoiceInformationExtraction\\GCN_based\\testResults_{time}.json", testResults)
            print("Average Test Loss: {}".format(overallLoss / len(dataset)))
            print("Average Test Accuracy: {}".format(overallAcc / len(dataset)))
            print("-" * 100)
            print("Testing of GCN-based complete")

            return testResults

    """
    Model Info:
    
    number of parameters: 108_310_272 
    """
