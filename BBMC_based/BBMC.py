import string
from transformers import BertTokenizerFast, BertModel
import torch
# from TorchCRF import CRF
from torchcrf import CRF
import pandas as pd
from utils.helperFunctions import loadJSON, getConfig, CONFIG_PATH, separate, createJSON
from datetime import datetime
from dataProcessing.customDataset import CustomDataset
from sklearn.metrics import accuracy_score, confusion_matrix

torch.manual_seed(123)

# Suppress the pandas append deprecation warning

TOKENIZER = BertTokenizerFast.from_pretrained('bert-base-cased')
MODEL = BertModel.from_pretrained('bert-base-cased')
tokenizer = TOKENIZER
modelCurrent = MODEL


def countTokensPerWord2(wordSeq: str, offsets: list) -> list:
    wordIndex = 0
    tokenCount = [0 for i in range(len(wordSeq.split()))]
    for count in range(len(offsets) - 1):
        tokenCount[wordIndex] += 1
        if offsets[count][1] != offsets[count + 1][0]:
            wordIndex += 1
    # with the increment-before-if-approach, the first one is counted one time too many
    tokenCount[0] -= 1
    return tokenCount


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


class InvoiceBBMC(torch.nn.Module):

    def __init__(self,
                 tokenizer=tokenizer,
                 model=modelCurrent,
                 hiddenDim=300,
                 LSTMlayers=100,
                 dropoutRate=.5,
                 numLabels=19,
                 batchSize=8,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        super(InvoiceBBMC, self).__init__()

        self.device = device
        self.batchSize = batchSize

        self.tokenizer = tokenizer
        self.bert = model.to(device)

        embeddingDim = self.bert.config.to_dict()['hidden_size']
        self.bilstm = torch.nn.LSTM(embeddingDim,
                                    hiddenDim // 2,
                                    num_layers=LSTMlayers,
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=dropoutRate
                                    ).to(device)

        self.multiheadAttention = torch.nn.MultiheadAttention(hiddenDim,
                                                              num_heads=3,
                                                              batch_first=True,
                                                              dropout=dropoutRate
                                                              ).to(device)

        self.fc1 = torch.nn.Linear(hiddenDim,
                                   numLabels
                                   ).to(device)
        self.crf = CRF(numLabels,
                        batch_first=True
                       ).to(device)

    def getSequence(self, dataInstance):
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

    def prepareInput(self, dataInstance):

        inputSeq = self.getSequence(dataInstance)
        tokenizedSeq = self.tokenizer(inputSeq, return_offsets_mapping=True, return_tensors="pt")

        return tokenizedSeq

    def forward(self, inputTensor, attentionMask, labels=None):

        outputs = self.bert(inputTensor, attention_mask=attentionMask)
        x = outputs[0]

        x, _ = self.bilstm(x)

        x = self.dropout(x)
        x, _ = self.multiheadAttention(x, x, x)

        emissions = self.fc1(x)

        if labels is not None:
            loss = -self.crf(emissions, labels, reduction='mean', mask=attentionMask.bool())
            tags = self.crf.decode(emissions, mask=attentionMask.bool())
            return loss, tags
        else:
            prediction = self.crf.decode(emissions, mask=attentionMask.bool())
            return -1, prediction

    def getGoldLabels(self, dataInstance, tokenizedSequence):

        dataSequenceAsList = self.getSequence(dataInstance).split(" ")
        dataSequence = "".join(dataSequenceAsList).lower()
        groundTruth = dataInstance["goldLabels"]

        goldLabelsChar = ["O" for _ in range(tokenizedSequence["input_ids"].size(1) - 2)]
        tokensAsText = self.tokenizer.convert_ids_to_tokens(tokenizedSequence["input_ids"].tolist()[0],
                                                            skip_special_tokens=True)

        labelTranslation = {f"{tag}-{i}": (2 * count + 1) + (counter * 1) for count, i in
                            enumerate(groundTruth.keys()) for
                            counter, tag in enumerate(["B", "I"])}
        labelTranslation["O"] = 0

        for tag, subDict in groundTruth.items():
            if subDict is not None:
                truthStringAsList = subDict["value"].split(" ")
                truthString = "".join(truthStringAsList).lower()
                try:
                    matchInString = dataSequence.index(truthString)
                    startIndex = 0
                    temp = 0
                    # smaller than instead of == so as to cover cases where the focal token is inside another token
                    while temp <= matchInString:
                        temp += len(dataSequenceAsList[startIndex])
                        startIndex += 1
                    if temp != matchInString:
                        startIndex -= 1
                    wordIndices = [startIndex]
                    temp = len(dataSequenceAsList[wordIndices[-1]])
                    while temp < len(truthString):
                        wordIndices.append(wordIndices[-1] + 1)
                        temp += len(dataSequenceAsList[wordIndices[-1]])

                    tokensPerWord = countTokensPerWord(" ".join(dataSequenceAsList),
                                                       tokenizedSequence.encodings[0].offsets[1:-1])
                    startIndex2 = sum(tokensPerWord[:wordIndices[0]])
                    tokenRange = sum([tokensPerWord[i] for i in wordIndices])
                    goldLabelsChar[startIndex2] = f"B-{tag}"
                    startIndex2 += 1
                    tokenRange -= 1
                    while tokenRange != 0:
                        goldLabelsChar[startIndex2] = f"I-{tag}"
                        startIndex2 += 1
                        tokenRange -= 1

                except ValueError:
                    wordIndices = 0

        goldLabels = [labelTranslation[i] for i in goldLabelsChar]
        return goldLabels, goldLabelsChar, labelTranslation

    def getGoldLabels2(self, dataInstance, feedback=False):

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

    def trainModel(self,
                   numEpochs,
                   dataset,
                   trainHistoryPath="",
                   lr=1e-3):

        if trainHistoryPath:
            trainHistory = pd.read_csv(trainHistoryPath)

        try:
            epochData = pd.read_csv("trainEpochData_06-06.csv")
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

                batchDataIndex = f"{epoch}_{i}"
                batchData[batchDataIndex] = {"batchLoss": 0,
                                             "batchItems": [],
                                             "goldLabels": [],
                                             "predictions": []
                                             }

                allInstances = shuffledIndices[i - batchSize:i]
                preparedInputList = []
                attentionMaskList = []
                labelsList = []

                # innermost loop - respectively handles concrete instances in each batch
                for batchNum, idx in enumerate(allInstances):

                    instance = dataset[idx]

                    pathToInstance = instance["instanceFolderPath"]
                    itemNum = pathToInstance.split("\\")[-1]

                    batchData[batchDataIndex]["batchItems"].append(itemNum)

                    print(i, batchNum, pathToInstance)

                    preparedInput = self.prepareInput(instance).to(self.device)

                    if trainHistoryPath and f"{itemNum}_{epoch}" in trainHistory.values:
                        continue

                    if trainHistoryPath:
                        trainHistory.loc[len(trainHistory)] = f"{itemNum}_{epoch}"

                    goldLabels, goldLabelsChar, labelTranslation = self.getGoldLabels(instance, preparedInput)
                    goldLabels = torch.tensor([goldLabels]).to(self.device)

                    preparedInputList.append(preparedInput["input_ids"][0])
                    labelsList.append(goldLabels[0])
                    attentionMaskList.append(torch.ones((1, preparedInput["input_ids"].size(1))))

                # end of innermost loop - i.e. pre-processing for all invoice instances of batch complete

                if not preparedInputList:
                    continue

                batchData[batchDataIndex]["goldLabels"] = [i.tolist() for i in labelsList]

                maxBatchLength = max(t.size(0) for t in preparedInputList)

                preparedInputList = torch.stack([torch.nn.functional.pad(t,
                                                                         (0,
                                                                          maxBatchLength - t.size(0))
                                                                         ) for t in preparedInputList],
                                                dim=0)

                labelsList = torch.stack([torch.nn.functional.pad(t,
                                                                  (0,
                                                                   maxBatchLength - t.size(0))
                                                                  ) for t in labelsList],
                                         dim=0)

                attentionMask = torch.stack([torch.nn.functional.pad(t,
                                                                     (0,
                                                                      maxBatchLength - t.size(1))
                                                                     ) for t in attentionMaskList],
                                            dim=0)

                attentionMask = attentionMask.view(attentionMask.size(0), attentionMask.size(2)).to(self.device)

                self.zero_grad()
                loss, predictions = self.forward(preparedInputList, attentionMask, labels=labelsList)

                batchData[batchDataIndex]["predictions"] = predictions

                overallEpochLoss += loss.item()
                batchData[batchDataIndex]["batchLoss"] = loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

            # end of intermediate loop - respective epoch complete

            createJSON(r"F:\CodeCopy\InvoiceInformationExtraction\BBMC_based\batchData.json", batchData)

            if trainHistoryPath:
                trainHistory.to_csv(trainHistoryPath, index=False)

            # after each epoch, save current state dict as checkpoint
            time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            checkpointPath = getConfig("BBMC_based", CONFIG_PATH)["pathToStateDict"][:-3] + f"{epoch}_"
            checkpointPath += time
            checkpointPath += ".pt"
            torch.save(self.state_dict(), checkpointPath)

            overallEpochLoss = overallEpochLoss / ((len(dataset) // self.batchSize) * self.batchSize)
            epochData = epochData.append({'epoch': epoch + 1, 'avgLoss': overallEpochLoss}, ignore_index=True)
            scheduler.step()

            epochData.to_csv(r"F:\CodeCopy\InvoiceInformationExtraction\BBMC_based\trainEpochData.csv", index=False)

        torch.save(self.state_dict(), getConfig("BBMC_based", CONFIG_PATH)["pathToStateDict"])
        print("Training of BBMC-based model complete")

    def testModel(self,
                  dataset
                  ):

        testResults = {}
        self.eval()

        with torch.no_grad():
            overallAcc = 0
            overallLoss = 0

            for c, idx in enumerate(range(len(dataset))):
                instance = dataset[idx]
                pathToInstance = instance["instanceFolderPath"]

                print(c, pathToInstance)

                itemNum = pathToInstance.split("\\")[-1]
                testResults[itemNum] = {}

                preparedInput = self.prepareInput(instance).to(self.device)
                testResults[itemNum]["instanceTokens"] = self.tokenizer.convert_ids_to_tokens(
                    preparedInput["input_ids"][0])

                goldLabels, goldLabelsChar, labelTranslation = self.getGoldLabels(instance, preparedInput)

                attentionMask = torch.ones((1, preparedInput["input_ids"].size(1))).to(self.device)

                preparedInput = preparedInput["input_ids"][0][None, :]

                # append labels for the special tokens
                goldLabels.append(0)
                goldLabels.insert(0, 0)
                goldLabels = torch.tensor([goldLabels]).to(self.device)

                loss, predictions = self.forward(preparedInput, attentionMask, labels=goldLabels)
                overallLoss += loss.item()

                testResults[itemNum]["goldLabels"] = goldLabels[0].tolist()
                testResults[itemNum]["predictions"] = predictions[0]
                testResults[itemNum]["loss"] = loss.item()
                testResults[itemNum]["NumberOfNonZeroLabelsPredicted"] = sum(
                    list(map(lambda x: 1 if x != 0 else 0, predictions[0])))
                testResults[itemNum]["NumberOfNonZeroLabelsGold"] = sum(
                    list(map(lambda x: 1 if x != 0 else 0, goldLabels[0].tolist())))

                testResults[itemNum]["accuracy"] = accuracy_score(goldLabels[0].tolist(),
                                                                  predictions[0])

                overallAcc += testResults[itemNum]["accuracy"]

                testResults[itemNum]["confusionMatrix"] = confusion_matrix(
                    goldLabels[0].tolist(),
                    predictions[0]).tolist()

            time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            createJSON(f"F:\\CodeCopy\\InvoiceInformationExtraction\\BBMC_based\\testResults{time}.json", testResults)
            print("Average Test Loss: {}".format(overallLoss / len(dataset)))
            print("Average Test Accuracy: {}".format(overallAcc / len(dataset)))
            print("-" * 100)
            print("Testing of BBMC complete")
            return testResults

    def testModel2(self, dataset):

        testResults = pd.DataFrame(
            columns=['invoiceInstance', 'prediction', "goldLabels", "goldLabelsChar", "labelTranslation",
                     "instanceLoss"])
        self.eval()

        with torch.no_grad():
            for i in range(len(dataset)):
                dataInstance = dataset[i]
                preparedInput = self.prepareInput(dataInstance)
                # preparedInput = preparedInput.type(dtype=torch.float32)

                goldLabels, goldLabelsChar, labelTranslation = self.getGoldLabels(dataInstance, preparedInput)
                goldLabels = torch.tensor([goldLabels])

                loss, predictions = self.forward(preparedInput, goldLabels)

                testResults = pd.concat([testResults,
                                         pd.Series([dataInstance["instanceFolderPath"], predictions, goldLabels,
                                                    goldLabelsChar,
                                                    labelTranslation, loss])])

            time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            testResults.to_csv(f"./testResults_{time}.csv")
            print("Testing of BBMC model complete")

        return testResults

    """
    Model info:
    
    - number of parameters: 163_479_190
    - time to train for 2,000 samples for one epoch:
        ~ 250 minutes
    """
