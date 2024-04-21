import torch
from datetime import datetime
from utils.helperFunctions import loadJSON, getConfig, separate, CONFIG_PATH
from dataProcessing.customDataset import CustomDataset
from CloudScan_based.featureExtraction import featureCalculation
import numpy as np
from sklearn.feature_extraction import FeatureHasher
import pandas as pd

# TODO: Change the append for the dataframes in the training and testing methods

"""
For the nGrammer, ngrams of maximum length 4 are derived. I.e., if 4-grams are possible for a focal
word, they are taken; alternatively, should only bigrams be possible, they are used.
The features are then calculated for each word on the basis of the ngram information.
These features are then encoded and in combination with the custom features fed to the word-level LSTM
"""


def check_sequence_existence(tokens_text, tokens_seq):
    for i in range(len(tokens_text) - len(tokens_seq) + 1):
        l1 = tokens_text[i:i + len(tokens_seq)]
        l1 = [j.lower() for j in l1]
        l2 = [j.lower() for j in tokens_seq]
        # print(l1,l2)
        if l1 == l2:
            return True, i  # Returns True and start index if sequence is found
    return False, -1  # Returns False and -1 if sequence is not found


class CloudScanLSTM(torch.nn.Module):

    def __init__(self, hashSize=2 ** 18, embeddingSize=500, inputSize=527, numLabels=19):
        super(CloudScanLSTM, self).__init__()

        self.hasher = FeatureHasher(n_features=hashSize, input_type='string')
        self.embedding = torch.nn.Embedding(hashSize, embeddingSize)

        self.feedforward1 = torch.nn.Sequential(
            torch.nn.Dropout(.5),
            torch.nn.Linear(inputSize, 600),
            torch.nn.ReLU(),
            torch.nn.Dropout(.5),
            torch.nn.Linear(600, 600),
            torch.nn.ReLU()
        )

        # self.Dropout = torch.nn.Dropout(.5)
        self.bilstm = torch.nn.LSTM(600, 400, 1, batch_first=True, bidirectional=True)

        self.feedforward2 = torch.nn.Sequential(
            torch.nn.Dropout(.5),
            torch.nn.Linear(800, 600),  # Adjusted for bidirectional LSTM output
            torch.nn.ReLU(),
            torch.nn.Dropout(.5),
            torch.nn.Linear(600, 600),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(600, numLabels),
            # torch.nn.Softmax(dim=1)
        )

    def nGrammer(self, dataInstance) -> list:
        nGrams = []

        # Get and format df with the underlying information
        featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
        colNames = list(featuresDF.columns)
        colNames[0] = "wordKey"
        featuresDF.columns = colNames

        # For each word in the df, calculate ngrams
        for word in featuresDF["wordKey"]:
            wordSet = ([word], word)

            contextCount = 1
            nextRightWord = featuresDF.loc[featuresDF["wordKey"] == word, "right"].item()
            while nextRightWord is not np.nan and contextCount < 4:
                wordSet[0].append(nextRightWord)
                nextRightWord = featuresDF.loc[featuresDF["wordKey"] == nextRightWord, "right"].item()
                contextCount += 1
            nGrams.append(wordSet)

        return nGrams

    def prepareInput(self, wordFeatures, useTextualFeatures=True):
        words = [i[0] for i in wordFeatures]
        textFeatures = [i[1] for i in wordFeatures]
        numericFeatures = [i[2] for i in wordFeatures]
        boolFeatures = [i[3] for i in wordFeatures]
        nGrams = [i[4] for i in wordFeatures]

        # As performed in the underlying paper, textual features are first hashed to a
        # 2**18 binary vector and then embedded
        if useTextualFeatures:
            binaryArray = list(map(lambda x: self.hasher.transform([[str(i) for i in x]]).toarray(),
                                   textFeatures))
        else:
            binaryArray = list(map(
                lambda x: self.hasher.transform([[x.split("_")[0]]]).toarray(),
                words))
        embeddingIndices = list(map(lambda x: x.nonzero()[1], binaryArray))

        overallEmbeddings = []
        for indexList in embeddingIndices:
            indexList = torch.tensor(indexList)
            wordEmbeddings = torch.mean(self.embedding(indexList), dim=0)
            overallEmbeddings.append(wordEmbeddings)
        numericTensor = torch.tensor(numericFeatures)
        numericTensor = torch.nn.functional.normalize(numericTensor, p=1.)
        boolTensor = torch.tensor(boolFeatures)

        preparedTensor = torch.stack(overallEmbeddings)
        preparedTensor = torch.concat([preparedTensor, numericTensor, boolTensor], dim=1)
        preparedTensor = preparedTensor.view(1, preparedTensor.size()[0], preparedTensor.size()[1])

        return preparedTensor, words

    def getGoldLabels(self, dataInstance, feedback=False):

        # Uses IOB-tagging scheme
        featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
        colNames = list(featuresDF.columns)
        colNames[0] = "wordKey"
        featuresDF.columns = colNames

        goldLabelsChar = ["O" for i in range(len(featuresDF))]
        groundTruth = dataInstance["goldLabels"]
        hOCRcharSeq = "".join(list(map(lambda x: x.split("_")[0], featuresDF["wordKey"])))
        groundTruthCharSeq = [i["value"].replace(" ", "") for i in groundTruth.values() if i is not None]
        b = (list(map(lambda x: x.split("_")[0], featuresDF["wordKey"])))
        labelTranslation = {f"{tag}-{i}": (2 * count + 1) + (counter * 1) for count, i in enumerate(groundTruth.keys()) for
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

    def forward(self, inputTensor):

        x = self.feedforward1(inputTensor)

        x, (hn, cn) = self.bilstm(x)
        # TODO: check whether dimensions fit
        x = self.feedforward2(x)
        result = self.classifier(x)

        return result

    def trainModel(self, numEpochs, dataset, lr=1e-5):

        epochData = pd.DataFrame(columns=['epoch', 'avgLoss'])
        batchData = pd.DataFrame(columns=['epoch', 'batch', 'loss'])

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)

        self.train()
        for epoch in range(numEpochs):
            print(f"Epoch {epoch + 1} / {numEpochs}")

            overallEpochLoss = 0
            shuffledIndices = torch.randperm(len(dataset))

            for i in range(len(dataset)):
                instance = dataset[shuffledIndices[i]]
                nGramList = self.nGrammer(instance)
                derivedFeatures = featureCalculation(nGramList, instance)
                preparedInput, words = self.prepareInput(derivedFeatures, useTextualFeatures=True)
                goldLabels, goldLabelsChar, labelTranslation = self.getGoldLabels(instance)
                goldLabels = torch.tensor([goldLabels])

                logits = self.forward(preparedInput)
                logits = logits[0]
                goldLabels = goldLabels[0]
                labels = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)

                self.zero_grad()

                loss = torch.nn.functional.cross_entropy(logits, goldLabels)
                batchData = batchData.append(
                    {'epoch': epoch + 1, 'batch': i + 1, 'loss': loss.item()},
                    ignore_index=True)
                loss.backward()
                optimizer.step()
                overallEpochLoss += loss.item()

            overallEpochLoss = overallEpochLoss / len(dataset)
            print(f"Avg. loss for epoch {epoch + 1}: {overallEpochLoss}")
            epochData = epochData.append({'epoch': epoch + 1, 'avg_loss': overallEpochLoss}, ignore_index=True)


        time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        epochData.to_csv(f"./trainEpochData_{time}.csv")
        batchData.to_csv(f"./trainBatchData_{time}.csv")
        print("Training of CloudScan-based model complete")

    def testModel(self, dataset):
        testResults = pd.DataFrame(columns=['invoiceInstance', 'prediction', "goldLabels", "instanceLoss"])

        self.eval()

        with torch.no_grad():
            for i in range(len(dataset)):
                dataInstance = dataset[i]
                nGramList = self.nGrammer(dataInstance)
                derivedFeatures = featureCalculation(nGramList, dataInstance)
                preparedInput, words = self.prepareInput(derivedFeatures, useTextualFeatures=True)
                goldLabels, goldLabelsChar, labelTranslation = self.getGoldLabels(dataInstance)
                goldLabels = torch.tensor([goldLabels])

                logits = self.forward(preparedInput)
                logits = logits[0]
                goldLabels = goldLabels[0]
                labels = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
                loss = torch.nn.functional.cross_entropy(logits, goldLabels)

                testResults = pd.concat(
                    [testResults, pd.Series([dataInstance["instanceFolderPath"], labels, goldLabels, loss.item()])])
        time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        testResults.to_csv(f"./testResults_{time}.csv")

        print("Testing of CloudScan-based model complete")
        return testResults


if __name__ == "__main__":
    data = CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
    cloudScan = CloudScanLSTM()
    #torch.save(cloudScan.state_dict(), getConfig("CloudScan_based", CONFIG_PATH)["pathToStateDict"])
    #cloudScan.load_state_dict(torch.load(getConfig("CloudScan_based", CONFIG_PATH)["pathToStateDict"]))
    # cloudScan.trainModel(2, data)
    cloudScan.testModel(data)
