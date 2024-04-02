import string
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric.data
from torch_geometric.nn.conv import ChebConv
from transformers.utils import logging
import pandas as pd
from transformers import BertTokenizerFast, BertModel
from dataProcessing.customDataset import CustomDataset
from utils.helperFunctions import getConfig, CONFIG_PATH
from GCN_based.featureExtraction import featureCalculation

# logging.set_verbosity_info()
TOKENIZER = BertTokenizerFast.from_pretrained('bert-base-cased')
MODEL = BertModel.from_pretrained('bert-base-cased')
tokenizer = TOKENIZER
model = MODEL


def countTokensPerWord(wordSeq: str, offsets: list) -> list:
    wordIndex = 0
    tokenCount = [0 for i in range(len(wordSeq.split()))]
    for count in range(len(offsets) - 1):
        tokenCount[wordIndex] += 1
        if offsets[count][1] != offsets[count + 1][0]:
            wordIndex += 1
    # with the increment-before-if-approach, the first one is counted one time too many
    tokenCount[0] -= 1
    return tokenCount


class InvoiceGCN(torch.nn.Module):

    def __init__(self, dataset, model=model, tokenizer=tokenizer):
        super(InvoiceGCN, self).__init__()

        self.dataset = dataset
        self.initFilterNumber = 16
        self.numClasses = 10

        self.tokenizer = tokenizer
        self.embeddings = model.embeddings

        self.relu = torch.nn.ReLU()
        self.gcn1 = ChebConv(785, self.initFilterNumber, K=3)
        self.gcn2 = ChebConv(self.initFilterNumber, 2 * self.initFilterNumber, K=3)
        self.gcn3 = ChebConv(2 * self.initFilterNumber, 4 * self.initFilterNumber, K=3)
        self.gcn4 = ChebConv(4 * self.initFilterNumber, 8 * self.initFilterNumber, K=3)
        self.gcn5 = ChebConv(8 * self.initFilterNumber, self.numClasses, K=3)
        self.softmax = torch.nn.Softmax()

    def getSequence(self, dataInstance):
        featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
        colNames = list(featuresDF.columns)
        colNames[0] = "wordKey"
        featuresDF.columns = colNames

        seqString = ""
        seqList = list(map(lambda x: x.split("_")[0], featuresDF["wordKey"]))

        seqString += seqList[0]
        punct = ['.', ',', ';', ':', '!', '?', '-', '_', '(', ')', '[', ']', '{', '}', '"', "'", '...', '–', '—',
                 '/',
                 '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '~', '`']

        for i in seqList[1:]:
            if i in punct:
                seqString += i
            else:
                seqString += f" {i}"

        return seqString

    def embedWords(self, wordSeq):

        tokens = tokenizer(wordSeq, return_offsets_mapping=True)
        offsets = tokens.encodings[0].offsets

        tokensPerWord = countTokensPerWord(wordSeq, offsets)

        tokenEmbs = self.embeddings(torch.tensor([tokens["input_ids"]]))

        wordEmbeds = []
        wordIndex = 0
        index = 0
        for i in tokensPerWord:
            wordEmbeds.append(tokenEmbs[0, wordIndex:wordIndex + i, :].mean(dim=0))
            index += i

        wordEmbeds = torch.stack(wordEmbeds)
        # wordEmbeds = wordEmbeds.view(1, wordEmbeds.size(0), wordEmbeds.size(1))
        return wordEmbeds

    def plotGraph(self, data):
        G = nx.Graph()
        edge_index = data.edge_index.cpu().numpy()
        pos = {i: (coords[0], abs(2200 - coords[1])) for i, coords in enumerate(data.pos.cpu().numpy())}
        edges = zip(edge_index[0], edge_index[1])
        G.add_edges_from(edges)
        wordLabels = {i: data.words[i] for i in range(len(data.words))}

        # Plot the graph
        nx.draw(G, pos, with_labels=False, node_size=70, node_color="blue", alpha=0.6, edge_color="gray")
        nx.draw_networkx_labels(G, pos, labels=wordLabels, font_size=9)  # Add word labels
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
        customFeatures = torch.tensor(customFeatures)
        nodeFeatures = torch.cat((wordEmbeddings, customFeatures), dim=1)
        nodeLabels, _, __ = self.getGoldLabels(dataInstance)
        nodeLabels = torch.tensor(nodeLabels, dtype=torch.long)

        return nodeWords, nodeFeatures, nodeLabels

    def getGoldLabels(self, dataInstance, feedback=False):

        featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
        colNames = list(featuresDF.columns)
        colNames[0] = "wordKey"
        featuresDF.columns = colNames

        goldLabelsChar = ["O" for i in range(len(featuresDF))]
        groundTruth = dataInstance["goldLabels"]
        hOCRcharSeq = "".join(list(map(lambda x: x.split("_")[0], featuresDF["wordKey"])))
        groundTruthCharSeq = [i["value"].replace(" ", "") for i in groundTruth.values() if i is not None]
        b = (list(map(lambda x: x.split("_")[0], featuresDF["wordKey"])))
        labelTranslation = {f"{tag}-{i}": count + 1 + counter for count, i in enumerate(groundTruth.keys()) for
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

    def getEdges(self, dataInstance):
        featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
        colNames = list(featuresDF.columns)
        colNames[0] = "wordKey"
        featuresDF.columns = colNames

        edgeList = []
        nodeNameToIndex = {}

        count = 0
        for i in featuresDF.loc[:, "wordKey"].tolist():
            if i.split("_")[0] not in string.punctuation:
                nodeNameToIndex[i] = count
                count += 1
        # nodeNameToIndex = {i:count for count, i in enumerate(self.getNodes(dataInstance)[0])}

        for word in featuresDF.loc[:, "wordKey"].tolist():
            if word.split("_")[0] in string.punctuation:
                continue
            for direction in ["left", "right", "above", "below"]:
                temp = featuresDF.loc[featuresDF["wordKey"] == word, direction].item()
                if temp is not np.nan:
                    edgeList.append([nodeNameToIndex[word],
                                     nodeNameToIndex[temp]])

        edgeTensor = torch.tensor(edgeList, dtype=torch.long)
        return edgeTensor

    def graphModeller(self, dataInstance):
        nodeWords, nodeFeatures, nodeLabels = self.getNodes(dataInstance)
        edgeIndex = self.getEdges(dataInstance)

        featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
        colNames = list(featuresDF.columns)
        colNames[0] = "wordKey"
        featuresDF.columns = colNames

        coords = [[int(i.split("_")[1]), int(i.split("_")[2])] for i in featuresDF.loc[:, "wordKey"].tolist()]
        coords = torch.tensor(coords)

        data = torch_geometric.data.Data(x=nodeFeatures,
                                         edge_index=edgeIndex.t().contiguous(),
                                         y=nodeLabels)
        data.words = nodeWords
        data.pos = coords

        return data

    def forward(self, graphData):

        nodes, edgeIndex = graphData.x, graphData.edge_index

        x = self.relu(self.gcn1(nodes, edgeIndex))
        x = self.relu(self.gcn2(x, edgeIndex))
        x = self.relu(self.gcn3(x, edgeIndex))
        x = self.relu(self.gcn4(x, edgeIndex))
        x = self.relu(self.gcn5(x, edgeIndex))

        return x

    def trainModel(self, numEpochs, dataset, lr=1e-3):

        resList = []

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)
        criterion = torch.nn.CrossEntropyLoss()

        self.train()
        for epoch in range(numEpochs):
            print(f"Epoch {epoch + 1} / {numEpochs}")
            overallLoss = 0
            shuffledIndices = torch.randperm(len(dataset))
            for i in range(len(dataset)):
                # get one specific invoice
                # That is to say, currently invoices are fed in batches of 1 to the algorithm
                dataInstance = dataset[shuffledIndices[i]]
                graphData = self.graphModeller(dataInstance)
                goldLabels = graphData.y

                self.zero_grad()
                x = self.forward(graphData)
                predictions = torch.argmax(x, dim=1)
                loss = criterion(x, goldLabels)
                resList.append((loss, predictions))
                loss.backward()
                optimizer.step()

                overallLoss += loss.item()
            print(overallLoss)
        return resList


if __name__ == '__main__':
    data = CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
    invoiceGCN = InvoiceGCN(data)

    invoiceGCN.trainModel(10, data)
