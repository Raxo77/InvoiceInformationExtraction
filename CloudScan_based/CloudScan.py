import torch
import warnings
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction import FeatureHasher
from dataProcessing.customDataset import CustomDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from CloudScan_based.featureExtraction import featureCalculation
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.helperFunctions import getConfig, CONFIG_PATH, loadJSON, createJSON

torch.manual_seed(123)

warnings.filterwarnings(action='ignore', message="The frame.append method is deprecated and will be removed from ")

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
        if l1 == l2:
            return True, i  # Returns True and start index if sequence is found
    return False, -1  # Returns False and -1 if sequence is not found


class CloudScanLSTM(torch.nn.Module):

    def __init__(self,
                 hashSize=2 ** 18,
                 embeddingSize=500,
                 inputSize=527,
                 numLabels=19,
                 citiesGazetteer=None,
                 batchSize=8,
                 countriesGazetteer=None,
                 ZIPgazetteer=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        super(CloudScanLSTM, self).__init__()

        self.device = device
        self.batchSize = batchSize

        self.hasher = FeatureHasher(n_features=hashSize, input_type='string')
        self.embedding = torch.nn.Embedding(hashSize, embeddingSize).to(self.device)

        self.citiesGazetteer = citiesGazetteer
        self.countriesGazetteer = countriesGazetteer
        self.ZIPgazetteer = ZIPgazetteer

        self.feedforward1 = torch.nn.Sequential(
            torch.nn.Dropout(.5),
            torch.nn.Linear(inputSize, 600),
            torch.nn.ReLU(),
            torch.nn.Dropout(.5),
            torch.nn.Linear(600, 600),
            torch.nn.ReLU()
        ).to(self.device)

        self.bilstm = torch.nn.LSTM(600, 400, 1, batch_first=True, bidirectional=True).to(self.device)

        self.feedforward2 = torch.nn.Sequential(
            torch.nn.Dropout(.5),
            torch.nn.Linear(800, 600),  # Adjusted for bidirectional LSTM output
            torch.nn.ReLU(),
            torch.nn.Dropout(.5),
            torch.nn.Linear(600, 600),
            torch.nn.ReLU()
        ).to(self.device)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(600, numLabels),
            # torch.nn.Softmax(dim=1)
        ).to(self.device)

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
            while not pd.isna(nextRightWord) and contextCount < 4:
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

        """
        As performed in the underlying paper, textual features are first hashed to a 2**18 binary vector and then 
        embedded in a trainable 500-dimensional representation using an embedding layer.
        Specifically, the four textual features (rawText, rawTextLastWord, rawTextWordLeft, rawTextTwoWordsLeft) are
        hashed into one joint binary array. Subsequently, nonzero elements of the hashing vector are then used
        as indices for the embedding layer. The final embedding of all textual features for one focal word is then the
        mean of the respective embedding vectors. 
        """

        if useTextualFeatures:
            binaryArray = list(map(lambda x: self.hasher.transform([[str(i) for i in x if i is not None]]).toarray(),
                                   textFeatures))
        else:
            binaryArray = list(map(
                lambda x: self.hasher.transform([[x.split("_")[0]]]).toarray(),
                words))
        embeddingIndices = list(map(lambda x: x.nonzero()[1], binaryArray))

        overallEmbeddings = []
        for indexList in embeddingIndices:
            indexList = torch.tensor(indexList).to(self.device)
            wordEmbeddings = torch.mean(self.embedding(indexList), dim=0)
            overallEmbeddings.append(wordEmbeddings)
        numericTensor = torch.tensor(numericFeatures)

        lInfNorm = torch.max(torch.abs(numericTensor))
        numericTensor = numericTensor / lInfNorm if lInfNorm != 0 else numericTensor
        numericTensor = numericTensor.to(self.device
                                         )
        boolTensor = torch.tensor(boolFeatures)
        boolTensor = boolTensor.to(self.device)

        preparedTensor = torch.stack(overallEmbeddings)
        preparedTensor = torch.concat([preparedTensor, numericTensor, boolTensor], dim=1)
        # preparedTensor = preparedTensor.view(1, preparedTensor.size(0), preparedTensor.size(1))

        return preparedTensor, words

    def getGoldLabels(self, dataInstance):

        # Uses IOB-tagging scheme
        featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
        colNames = list(featuresDF.columns)
        colNames[0] = "wordKey"
        featuresDF.columns = colNames

        goldLabelsChar = ["O"] * len(featuresDF)
        groundTruth = dataInstance["goldLabels"]
        hOCRcharSeq = "".join(list(map(lambda x: x.split("_")[0], featuresDF["wordKey"])))
        groundTruthCharSeq = [i["value"].replace(" ", "") for i in groundTruth.values() if i is not None]
        b = (list(map(lambda x: x.split("_")[0], featuresDF["wordKey"])))
        labelTranslation = {f"{tag}-{i}": (2 * count + 1) + (counter * 1) for count, i in enumerate(groundTruth.keys())
                            for
                            counter, tag in enumerate(["B", "I"])}
        labelTranslation["O"] = 0

        """
        Mapping ground truth values to the tokens identified via hOCR: Since the subsequent model uses the hOCR text as
        input, but ground truth labels are evidently based on the vanilla/ground truth text, "different token
        arrangements" may arise. As such, each ground truth value is sought in the whitespace-free hOCR text; if found,
        the tokens of the original hOCR text are matched tagged in IOB scheme via the length of the ground truth value
        """
        for label, i in zip([i for i in groundTruth.keys() if groundTruth[i] is not None], groundTruthCharSeq):
            a = hOCRcharSeq.find(i)
            if a != -1:
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

        goldLabels = [labelTranslation[i] for i in goldLabelsChar]
        return goldLabels, goldLabelsChar, labelTranslation

        # For each gold label, match  char sequence of hOCR text with char sequence of groundTruth
        # This is done to circumnavigate issues arising from different "tokenization" of hOCR and groundTruth strings

    def forward(self, inputTensor):

        x = self.feedforward1(inputTensor)

        x, (hn, cn) = self.bilstm(x)

        x = self.feedforward2(x)
        result = self.classifier(x)

        return result

    def trainModel(self,
                   numEpochs,
                   dataset,
                   trainHistoryPath="",
                   lr=1e-5):

        if trainHistoryPath:
            trainHistory = pd.read_csv(trainHistoryPath)

        try:
            epochData = pd.read_csv("./trainEpochData.csv")
        except FileNotFoundError:
            epochData = pd.DataFrame(columns=['epoch', 'avgLoss'])

        try:
            batchData = loadJSON(f"./batchData.json")
        except FileNotFoundError:
            batchData = {}

        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)

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
                preparedInputList = []
                labelsList = []
                seqLenList = []

                # innermost loop - respectively handles concrete instances in each batch
                # due to their complexity/scope pre-processing tasks are performed per invoice
                for batchNum, idx in enumerate(allInstances):

                    dataInstance = dataset[idx]

                    pathToInstance = dataInstance["instanceFolderPath"]
                    itemNum = pathToInstance.split("\\")[-1]

                    batchData[batchDataIndex]["batchItems"].append(itemNum)

                    print(i, batchNum, pathToInstance)

                    if trainHistoryPath and f"{itemNum}_{epoch}" in trainHistory.values:
                        continue

                    if trainHistoryPath:
                        trainHistory.loc[len(trainHistory)] = f"{itemNum}_{epoch}"

                    nGramList = self.nGrammer(dataInstance)
                    derivedFeatures = featureCalculation(nGramList, dataInstance, citiesGazetteer=self.citiesGazetteer,
                                                         countryGazetteer=self.countriesGazetteer,
                                                         ZIPCodesGazetteer=self.ZIPgazetteer)
                    preparedInput, words = self.prepareInput(derivedFeatures, useTextualFeatures=True)
                    preparedInput = preparedInput.to(self.device)

                    goldLabels, goldLabelsChar, labelTranslation = self.getGoldLabels(dataInstance)
                    goldLabels = torch.tensor(goldLabels).to(self.device)

                    preparedInputList.append(preparedInput)
                    labelsList.append(goldLabels)
                    seqLenList.append(preparedInput.size(0))

                # end of innermost loop - i.e. pre-processing for all invoice instances of batch complete

                if not preparedInputList:
                    continue

                batchData[batchDataIndex]["goldLabels"] = [i.tolist() for i in labelsList]

                maxBatchLength = max(t.size(0) for t in preparedInputList)

                preparedInputList = torch.stack([torch.nn.functional.pad(t,
                                                                         (0,
                                                                          0,
                                                                          0,
                                                                          maxBatchLength - t.size(0))
                                                                         ) for t in preparedInputList],
                                                dim=0)

                optimizer.zero_grad()
                logits = self.forward(preparedInputList)

                predictedLabels = torch.argmax(torch.nn.functional.softmax(logits, dim=2), dim=2)

                batchData[batchDataIndex]["predictions"].append(predictedLabels.tolist())

                flattenedGoldLabels = torch.cat(labelsList)
                flattenedLabels = predictedLabels[0, 0:seqLenList[0]]
                flattenedLogits = logits[0, 0:seqLenList[0], :]
                for c, j in enumerate(seqLenList[1:]):
                    flattenedLogits = torch.cat((flattenedLogits, logits[c + 1, 0:j, :]))
                    flattenedLabels = torch.cat((flattenedLabels, predictedLabels[c + 1, 0:j]))

                # Consider rescaling the class weights via "weights" parameter (numClasses-long vector)
                loss = torch.nn.functional.cross_entropy(flattenedLogits, flattenedGoldLabels)
                overallEpochLoss += loss.item()
                batchData[batchDataIndex]["batchLoss"] = loss.item()

                loss.backward()
                optimizer.step()

            # end of intermediate loop - respective epoch complete

            createJSON(r"F:\CodeCopy\InvoiceInformationExtraction\CloudScan_based\batchData.json", batchData)

            if trainHistoryPath:
                trainHistory.to_csv(trainHistoryPath, index=False)

            # after each epoch, save current state dict as checkpoint
            time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            checkpointPath = getConfig("CloudScan_based", CONFIG_PATH)["pathToStateDict"][:-3] + f"{epoch}_"
            checkpointPath += time
            checkpointPath += ".pt"
            torch.save(self.state_dict(), checkpointPath)

            overallEpochLoss = overallEpochLoss / ((len(dataset) // self.batchSize) * self.batchSize)
            epochData = epochData.append({'epoch': epoch + 1, 'avgLoss': overallEpochLoss}, ignore_index=True)
            scheduler.step()

            epochData.to_csv(r"F:\CodeCopy\InvoiceInformationExtraction\CloudScan_based\trainEpochData.csv",
                             index=False)

        torch.save(self.state_dict(), getConfig("CloudScan_based", CONFIG_PATH)["pathToStateDict"])
        print("Training of CloudScan-based model complete")

    def testModel(self,
                  dataset
                  ):

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

                nGramList = self.nGrammer(dataInstance)
                testResults[itemNum]["instanceTokens"] = list(map(lambda x: x[1].split("_")[0], nGramList))

                derivedFeatures = featureCalculation(nGramList, dataInstance, citiesGazetteer=self.citiesGazetteer,
                                                     countryGazetteer=self.countriesGazetteer,
                                                     ZIPCodesGazetteer=self.ZIPgazetteer)
                preparedInput, words = self.prepareInput(derivedFeatures, useTextualFeatures=True)
                preparedInput = preparedInput[None, :, :].to(self.device)

                goldLabels, goldLabelsChar, labelTranslation = self.getGoldLabels(dataInstance)
                goldLabels = torch.tensor(goldLabels).to(self.device)

                logits = self.forward(preparedInput)

                predictedLabels = torch.argmax(torch.nn.functional.softmax(logits, dim=2), dim=2)

                flattenedGoldLabels = goldLabels
                flattenedLogits = logits[0, :, :]

                # Consider rescaling the class weights via "weights" parameter (numClasses-long vector)
                loss = torch.nn.functional.cross_entropy(flattenedLogits, flattenedGoldLabels)
                overallLoss += loss.item()

                testResults[itemNum]["goldLabels"] = goldLabels.tolist()
                testResults[itemNum]["predictions"] = predictedLabels[0].tolist()
                testResults[itemNum]["loss"] = loss.item()
                testResults[itemNum]["NumberOfNonZeroLabelsPredicted"] = sum(
                    list(map(lambda x: 1 if x != 0 else 0, predictedLabels[0].tolist())))
                testResults[itemNum]["NumberOfNonZeroLabelsGold"] = sum(
                    list(map(lambda x: 1 if x != 0 else 0, goldLabels.tolist())))

                testResults[itemNum]["accuracy"] = accuracy_score(goldLabels.tolist(),
                                                                  predictedLabels[0].tolist())

                overallAcc += testResults[itemNum]["accuracy"]

                testResults[itemNum]["confusionMatrix"] = confusion_matrix(
                    goldLabels.tolist(),
                    predictedLabels[0].tolist()).tolist()

            time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            createJSON(f"F:\\CodeCopy\\InvoiceInformationExtraction\\CloudScan_based\\testResults_{time}.json",
                       testResults)
            print("Average Test Loss: {}".format(overallLoss / len(dataset)))
            print("Average Test Accuracy: {}".format(overallAcc / len(dataset)))
            print("-" * 100)
            print("Testing of CloudScan complete")
            return testResults
