import nltk
import torch
import string
import torchtext
import pandas as pd
from torchcrf import CRF
#from TorchCRF import CRF
from nltk.util import ngrams
from datetime import datetime
from collections import OrderedDict
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.helperFunctions import getConfig, CONFIG_PATH, loadJSON, createJSON

torch.manual_seed(123)

# CONSTANTS
CONFIG_PATH = CONFIG_PATH


def getPrintableChars() -> list:
    """
    Create a list of all printable Latin-1 characters for the creation of the character embedding vocabulary
    :return: list of the respective characters
    """
    # based on ISO/IEC 8859-1
    printableChars = []
    for i in range(32, 127):
        printableChars.append(chr(i))
    for i in range(160, 256):
        printableChars.append(chr(i))

    return printableChars


def getGoldData(path: str):
    goldData = loadJSON(path)
    temp = goldData
    goldData = {}
    for key, value in temp.items():
        if value is not None:
            goldData[key] = value["value"]
        else:
            goldData[key] = None
    return goldData


def createVocab(charList):
    vocab = torchtext.vocab.vocab(OrderedDict([(char, 1) for char in charList]), specials=["<unk>", "<eos>"])
    vocab.set_default_index(len(charList))
    return vocab




class Invoice_BiLSTM_CNN_CRF(torch.nn.Module):

    def __init__(self,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 wordEmbeddingVectorsPath=getConfig("pathToFastTextVectors", CONFIG_PATH),
                 wordEmbeddingStoiPath=getConfig("pathToFastTextStoi", CONFIG_PATH),
                 maxWordsPerInvoice=512,
                 charEmbeddingSize=30,
                 kernelSize=3,
                 trainableEmbeddings=False,
                 batchSize=32
                 ):
        super(Invoice_BiLSTM_CNN_CRF, self).__init__()

        self.batchSize = batchSize
        self.device = device

        # Char Embedding:
        self.charVocab = createVocab(getPrintableChars())
        charVocabSize = len(self.charVocab)
        self.charEmbeddingSize = charEmbeddingSize
        self.maxWordsPerInvoice = maxWordsPerInvoice
        self.maxCharsPerWord = 25
        self.charEmbedding = torch.nn.Embedding(charVocabSize, charEmbeddingSize)

        # Conv1D Layer for char embedding
        self.conv1d = torch.nn.Conv1d(in_channels=charEmbeddingSize,
                                      out_channels=charEmbeddingSize,
                                      kernel_size=kernelSize,
                                      padding="same"
                                      ).to(device)

        self.charLSTM = torch.nn.LSTM(charEmbeddingSize, 50, num_layers=1).to(device)

        # Word Embedding:
        vectors = torch.load(wordEmbeddingVectorsPath)
        self.vectors = vectors
        self.embeddingStoi = torch.load(wordEmbeddingStoiPath)
        self.wordEmbedding = torch.nn.Embedding.from_pretrained(vectors, freeze=not trainableEmbeddings)
        self.embeddingForm = ("glove" * ("glove" in wordEmbeddingVectorsPath.lower())) + (
                "fastText" * ("fasttext" in wordEmbeddingVectorsPath.lower()))

        # BiLSTM-CRF
        self.bilstm = torch.nn.LSTM(self.vectors.size(1) + 50, 512 // 2, bidirectional=True,
                                    batch_first=True).to(device)
        self.fc1 = torch.nn.Conv1d(in_channels=512, out_channels=19, kernel_size=3, padding=((3 - 1) // 2),
                                   stride=1).to(device)
        # self.fc1 = torch.nn.Linear(512, 19)

        self.crf = CRF(19, batch_first=True).to(device)

        nltk.download("punkt")

    def getSequence(self, dataInstance) -> str:

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

    def padSequence(self, seq: list, targetLen, padToken="<eos>"):
        paddedSeq = seq[:targetLen]
        paddedSeq += [padToken] * (targetLen - len(paddedSeq))
        return paddedSeq

    def embedChars(self, wordSeq: str):
        wordSeq = wordSeq.split(" ")
        wordSeq = self.padSequence(wordSeq, self.maxWordsPerInvoice)
        wordSeq = list(map(lambda x: self.padSequence(list(x), self.maxCharsPerWord), wordSeq))
        batchIdxs = list(map(lambda word: self.charVocab(list(word)), wordSeq))

        batchIdxs = torch.tensor(data=batchIdxs)
        charEmbeds = self.charEmbedding(batchIdxs)

        return charEmbeds

    def embedWords(self, wordSeq: str):

        wordSeq = self.padSequence(wordSeq.split(" "), self.maxWordsPerInvoice)
        wordEmbeddings = []
        if self.embeddingForm == "glove":
            # As GloVe knows only whole words for embeddings, the entire sequence is split and each word in the
            # split sequence is embedded - granted a corresponding embedding exists in the vocab - or assigned the
            # OOV vector
            for word in wordSeq:
                word = word.lower()
                if word == "<eos>":
                    wordEmbeddings.append(torch.full((1, self.vectors.size(1)), -1))
                elif word in self.embeddingStoi:
                    wordEmbeddings.append(self.wordEmbedding(torch.tensor([self.embeddingStoi[word]])))
                else:
                    temp = torch.zeros([self.vectors.size(1)])
                    temp = temp.view(1, -1)
                    wordEmbeddings.append(temp)

        else:
            for word in wordSeq:
                word = word.lower()
                if word == "<eos>":
                    wordEmbeddings.append(torch.full((1, self.vectors.size(1)), -1))
                elif word in self.embeddingStoi:
                    wordEmbeddings.append(self.wordEmbedding(torch.tensor([self.embeddingStoi[word]])))
                else:
                    subwords = [''.join(gram) for gram in ngrams(word, 3)]
                    subwordEmbeddings = []
                    for subword in subwords:
                        if subword in self.embeddingStoi:
                            subwordEmbeddings.append(self.wordEmbedding(torch.tensor([self.embeddingStoi[subword]])))
                        else:
                            temp = torch.zeros([self.vectors.size(1)])
                            temp = temp.view(1, -1)
                            subwordEmbeddings.append(temp)
                    if not len(subwords):
                        subwordEmbeddings = [torch.zeros([self.vectors.size(1)]).view(1, -1)]
                    subwordEmbeddings = torch.stack(subwordEmbeddings).mean(dim=0)
                    wordEmbeddings.append(subwordEmbeddings)

        wordEmbeddings = torch.stack(wordEmbeddings)
        wordEmbeddings = wordEmbeddings.view(-1, self.vectors.size(1))
        return wordEmbeddings

    def prepareInput(self, dataInstance):

        invoiceSeq = self.getSequence(dataInstance)
        numPadded = len(invoiceSeq.split(" "))
        charEmbeddings = self.embedChars(invoiceSeq)
        charEmbeddings = torch.permute(charEmbeddings, (0, 2, 1)).to(self.device)
        charEmbeddings = self.conv1d(charEmbeddings)
        charEmbeddings = torch.permute(charEmbeddings, (0, 2, 1))
        charEmbeddings, _ = self.charLSTM(charEmbeddings)
        charEmbeddings = charEmbeddings[:, -1, :]

        wordEmbeddings = self.embedWords(invoiceSeq).to(self.device)

        preparedInput = torch.cat([wordEmbeddings, charEmbeddings], dim=1)
        numPadded = self.maxWordsPerInvoice - numPadded
        return preparedInput, numPadded

    def forward(self, inputTensor, labels=None, attentionMask=None):

        x, _ = self.bilstm(inputTensor)

        emissions = self.fc1(x.permute(0, 2, 1))
        emissions = emissions.permute(0, 2, 1)

        attentionMask = attentionMask.byte()

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attentionMask, reduction='mean')
            tags = self.crf.decode(emissions, mask=attentionMask)
            return loss, tags
        else:
            prediction = self.crf.decode(emissions, mask=attentionMask)
            return -1, prediction

    def getGoldLabels(self, dataSeq, dataInstance, numPadded):

        dataSeq = dataSeq.lower()
        dataSeqAsList = dataSeq.split(" ")
        dataSeq = "".join(dataSeqAsList)

        goldData = dataInstance["goldLabels"]

        attentionMask = [1 for i in range(self.maxWordsPerInvoice)]
        for i in range(numPadded):
            attentionMask[-(i + 1)] = 0
        attentionMask = torch.tensor([attentionMask])

        labelTranslation = {f"{tag}-{i}": (2 * count + 1) + (counter * 1) for count, i in enumerate(goldData.keys()) for
                            counter, tag in enumerate(["B", "I"])}
        labelTranslation["O"] = 0

        goldLabelsChar = ["O" for _ in range(self.maxWordsPerInvoice)]

        for tag, subDict in goldData.items():
            if subDict is not None:
                truthStringAsList = subDict["value"].split(" ")
                truthString = "".join(truthStringAsList).lower()

                matchInString = dataSeq.find(truthString)

                if matchInString != -1:
                    startIndex = 0
                    temp = 0
                    while temp <= matchInString:
                        temp += len(dataSeqAsList[startIndex])
                        startIndex += 1
                    if temp != matchInString:
                        startIndex -= 1
                    wordIndices = [startIndex]
                    temp = len(dataSeqAsList[wordIndices[-1]])
                    while temp < len(truthString):
                        wordIndices.append(wordIndices[-1] + 1)
                        temp += len(dataSeqAsList[wordIndices[-1]])
                    goldLabelsChar[wordIndices[0]] = f"B-{tag}"
                    for i in wordIndices[1:]:
                        goldLabelsChar[i] = f"I-{tag}"
                else:
                    wordIndices = 0

        goldLabels = [labelTranslation[i] for i in goldLabelsChar]
        goldLabels = torch.tensor([goldLabels])
        return goldLabels, goldLabelsChar, labelTranslation, attentionMask

    def trainModel(self,
                   dataset,
                   numEpochs=40,
                   trainHistoryPath="",
                   lr=1e-3):

        print("Training of BiLSTM_CNN_CRF initiated")

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

                batchDataIndex = f"{epoch}_{i}"
                batchData[batchDataIndex] = {"batchLoss": 0,
                                             "batchItems": [],
                                             "goldLabels": [],
                                             "predictions": []
                                             }

                allInstances = shuffledIndices[i - batchSize:i]
                preparedInputList = []
                labelsList = []
                attentionMaskList = []

                # innermost loop - respective data pre-processing for each data entry
                for batchNum, idx in enumerate(allInstances):

                    instance = dataset[idx]

                    pathToInstance = instance["instanceFolderPath"]
                    itemNum = pathToInstance.split("\\")[-1]

                    batchData[batchDataIndex]["batchItems"].append(itemNum)

                    print(i, batchNum, pathToInstance)

                    if trainHistoryPath and f"{itemNum}_{epoch}" in trainHistory.values:
                        continue

                    if trainHistoryPath:
                        trainHistory.loc[len(trainHistory)] = f"{itemNum}_{epoch}"

                    # dim of prepared input (512, 350)
                    # |--> 0th dimension is the padded input sequence
                    # |--> 1st dimension is the feature size
                    preparedInput, numPadded = self.prepareInput(instance)

                    dataSeq = self.getSequence(instance)

                    goldLabels, goldLabelsChar, labelTranslation, attentionMask = self.getGoldLabels(dataSeq,
                                                                                                     instance,
                                                                                                     numPadded)
                    goldLabels.to(self.device)

                    batchData[batchDataIndex]["goldLabels"].append(goldLabels[0, 0:torch.sum(attentionMask)].tolist())

                    preparedInputList.append(preparedInput)
                    labelsList.append(goldLabels[0])

                    attentionMaskInstance = torch.ones((1, 512))
                    attentionMaskInstance[0, -numPadded:] = 0
                    attentionMaskList.append(attentionMaskInstance[0])

                # end of innermost loop - i.e. pre-processing for all invoice instances of batch complete

                if not preparedInputList:
                    print("skipped")
                    continue

                preparedInputList = torch.stack(preparedInputList, dim=0)

                labelsList = torch.stack(labelsList, dim=0)
                labelsList = labelsList.to(self.device)

                attentionMask = torch.stack(attentionMaskList, dim=0).to(self.device)

                optimizer.zero_grad()
                loss, predictions = self.forward(preparedInputList, labelsList, attentionMask)

                batchData[batchDataIndex]["predictions"] = predictions

                overallEpochLoss += loss.item()
                batchData[batchDataIndex]["batchLoss"] = loss.item()

                loss.backward()
                optimizer.step()

            # end of intermediate loop - respective epoch complete

            createJSON(r"F:\CodeCopy\InvoiceInformationExtraction\BiLSTM_CNN_CRF_based\batchData.json", batchData)

            if trainHistoryPath:
                trainHistory.to_csv(trainHistoryPath, index=False)

            # after each epoch, save current state dict as checkpoint
            time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            checkpointPath = getConfig("BiLSTM_CNN_CRF_based", CONFIG_PATH)["pathToStateDict"][:-3] + f"{epoch}_"
            checkpointPath += time
            checkpointPath += ".pt"
            torch.save(self.state_dict(), checkpointPath)

            overallEpochLoss = overallEpochLoss / ((len(dataset) // self.batchSize) * self.batchSize)
            epochData = epochData.append({'epoch': epoch + 1, 'avgLoss': overallEpochLoss}, ignore_index=True)
            scheduler.step()

            epochData.to_csv(r"F:\CodeCopy\InvoiceInformationExtraction\BiLSTM_CNN_CRF_based\trainEpochData.csv", index=False)

        torch.save(self.state_dict(), getConfig("BiLSTM_CNN_CRF_based", CONFIG_PATH)["pathToStateDict"])
        print("Training of BiLSTM_CNN_CRF complete")

    def testModel(self,
                  dataset
                  ):

        """
        The test results will be stored in a json.
        Concretely, for each invoice item the json contains the
        *) itemNumber
        *) the respective gold labels
        *) the respective predictions
        *) loss
        *)
        """
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

                preparedInput, numPadded = self.prepareInput(instance)
                preparedInput = preparedInput[None, :, :]

                dataSeq = self.getSequence(instance)

                goldLabels, goldLabelsChar, labelTranslation, attentionMask = self.getGoldLabels(dataSeq,
                                                                                                 instance,
                                                                                                 numPadded)
                goldLabels = goldLabels.to(self.device)
                attentionMask = attentionMask.to(self.device)

                loss, predictions = self.forward(preparedInput, goldLabels, attentionMask)
                overallLoss += loss.item()

                testResults[itemNum]["goldLabels"] = goldLabels[0, 0:torch.sum(attentionMask)].tolist()
                testResults[itemNum]["predictions"] = predictions[0]
                testResults[itemNum]["loss"] = loss.item()
                testResults[itemNum]["NumberOfNonZeroLabelsPredicted"] = sum(
                    list(map(lambda x: 1 if x != 0 else 0, predictions[0])))
                testResults[itemNum]["NumberOfNonZeroLabelsGold"] = sum(
                    list(map(lambda x: 1 if x != 0 else 0, goldLabels[0, 0:torch.sum(attentionMask)].tolist())))

                testResults[itemNum]["accuracy"] = accuracy_score(goldLabels[0, 0:torch.sum(attentionMask)].tolist(),
                                                                  predictions[0])

                overallAcc += testResults[itemNum]["accuracy"]

                testResults[itemNum]["confusionMatrix"] = confusion_matrix(
                    goldLabels[0, 0:torch.sum(attentionMask)].tolist(),
                    predictions[0]).tolist()

        time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        createJSON(f"F:\\CodeCopy\\InvoiceInformationExtraction\\BiLSTM_CNN_CRF_based\\testResults{time}.json", testResults)
        print("Average Test Loss: {}".format(overallLoss / len(dataset)))
        print("Average Test Accuracy: {}".format(overallAcc / len(dataset)))
        print("-" * 100)
        print("Testing of BiLSTM_CNN_CRF complete")

        return testResults

    def testModel2(self, dataset):
        testResults = pd.DataFrame(columns=['invoiceInstance', 'prediction', "goldLabels", "instanceLoss"])
        self.eval()

        with torch.no_grad():
            for i in range(len(dataset)):
                instance = dataset[i]
                preparedInput, numPadded = self.prepareInput(instance)
                dataSeq = self.getSequence(instance)
                pathToInstance = instance["instanceFolderPath"]

                goldLabels, goldLabelsChar, labelTranslation, attentionMask = self.getGoldLabels(dataSeq, instance,
                                                                                                 numPadded)

                attentionMask = torch.tensor(attentionMask)
                loss, predictions = self.forward(preparedInput, goldLabels, attentionMask)
                testResults = pd.concat(
                    [testResults, [instance["instanceFolderPath"], predictions, goldLabels, loss]])

        time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        testResults.to_csv(f"./testResults_{time}.csv")

        print("Testing of BiLSTM_CNN_CRF complete")
        return testResults


    """"
    Model info:
    
    _With FastText_:
    - Number of parameters (also includes non-trainable/requires_grad = False): 757_110_706
    - NUmber of parameters (trainable only(requires_grad = True): 1_299_706
    
    _With GloVe_:
    - n.a.
    
    Time to train:
        For 81 invoices ~1 min.
    """

