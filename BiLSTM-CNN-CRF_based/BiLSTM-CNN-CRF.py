import nltk
import torch
import string
import torchtext
import pandas as pd
from torchcrf import CRF
# from TorchCRF import CRF
from nltk.util import ngrams
from datetime import datetime
from collections import OrderedDict
from dataProcessing.customDataset import CustomDataset
from utils.helperFunctions import getConfig, CONFIG_PATH, loadJSON

# CONSTANTS
CONFIG_PATH = CONFIG_PATH


def getPrintableChars() -> list:
    """
    Create a list of all printable ASCII characters for the creation of the character embedding vocabulary
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
    return vocab


"""
For the max length of words 40 is taken as initial estimate
"""


class Invoice_BiLSTM_CNN_CRF(torch.nn.Module):

    def __init__(self,
                 dataset,
                 wordEmbeddingVectorsPath=getConfig("pathToFastTextVectors", CONFIG_PATH),
                 wordEmbeddingStoiPath=getConfig("pathToFastTextStoi", CONFIG_PATH),
                 maxWordsPerInvoice=512,
                 charEmbeddingSize=30,
                 kernelSize=3,
                 trainableEmbeddings=False,
                 ):
        super(Invoice_BiLSTM_CNN_CRF, self).__init__()
        self.dataset = dataset

        # Char Embedding:
        self.charVocab = createVocab(getPrintableChars())
        charVocabSize = len(self.charVocab)
        self.charEmbeddingSize = charEmbeddingSize
        self.maxWordsPerInvoice = maxWordsPerInvoice
        self.maxCharsPerWord = 25
        self.charEmbedding = torch.nn.Embedding(charVocabSize, charEmbeddingSize)

        # Conv1D Layer for char embedding
        self.conv1d = torch.nn.Conv1d(in_channels=charEmbeddingSize, out_channels=charEmbeddingSize,
                                      kernel_size=kernelSize,
                                      padding="same")
        self.charLSTM = torch.nn.LSTM(charEmbeddingSize, 50, num_layers=1)

        # Word Embedding:
        vectors = torch.load(wordEmbeddingVectorsPath)
        self.vectors = vectors
        self.embeddingStoi = torch.load(wordEmbeddingStoiPath)
        self.wordEmbedding = torch.nn.Embedding.from_pretrained(vectors, freeze=trainableEmbeddings)
        self.embeddingForm = ("glove" * ("glove" in wordEmbeddingVectorsPath.lower())) + (
                "fastText" * ("fasttext" in wordEmbeddingVectorsPath.lower()))

        # BiLSTM-CRF
        self.bilstm = torch.nn.LSTM(self.vectors.size(1) + charEmbeddingSize, 512 // 2, bidirectional=True)
        self.fc1 = torch.nn.Conv1d(in_channels=512, out_channels=19, kernel_size=3, padding=((3 - 1) // 2), stride=1)
        # self.fc1 = torch.nn.Linear(512, 19)
        self.crf = CRF(19, batch_first=True)

    def getSequence(self, dataInstance) -> str:

        # As in the context of this model no tokenizer is employed, the input features for this model do not
        # contain punctuation
        featuresDF = pd.read_csv(dataInstance["BERT-basedNoPunctFeaturesPath"])
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
            nltk.download("punkt")
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
        charEmbeddings = torch.permute(charEmbeddings, (0, 2, 1))
        charEmbeddings = self.conv1d(charEmbeddings)
        charEmbeddings = torch.permute(charEmbeddings, (0, 2, 1))
        charEmbeddings, _ = self.charLSTM(charEmbeddings)
        charEmbeddings = charEmbeddings[:, -1, :]

        wordEmbeddings = self.embedWords(invoiceSeq)

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
            attentionMask[(i + 1) * -1] = 0
        attentionMask = torch.tensor([attentionMask])

        labelTranslation = {f"{tag}-{i}": (2 * count + 1) + (counter * 1) for count, i in enumerate(goldData.keys()) for
                            counter, tag in enumerate(["B", "I"])}
        labelTranslation["O"] = 0

        goldLabelsChar = ["O" for i in range(self.maxWordsPerInvoice)]

        for tag, subDict in goldData.items():
            if subDict is not None:
                truthStringAsList = subDict["value"].split(" ")
                truthString = "".join(truthStringAsList).lower()
                try:
                    matchInString = dataSeq.index(truthString)
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


                except ValueError:
                    wordIndices = 0

        goldLabels = [labelTranslation[i] for i in goldLabelsChar]
        goldLabels = torch.tensor([goldLabels])
        return goldLabels, goldLabelsChar, labelTranslation, attentionMask

    def trainModel(self, dataset, numEpochs=40, trainHistoryPath="", lr=1e-3):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)

        epochData = pd.DataFrame(columns=['epoch', 'avgLoss'])
        batchData = pd.DataFrame(columns=['epoch', 'batch', 'loss'])

        if trainHistoryPath:
            trainHistory = pd.read_csv(trainHistoryPath)

        self.train()
        for epoch in range(numEpochs):
            print(f"Epoch {epoch + 1} / {numEpochs}")
            shuffledIndices = torch.randperm(len(dataset))

            overallEpochLoss = 0
            shuffledIndices = torch.randperm(len(dataset))
            for i in range(len(dataset)):
                instance = dataset[shuffledIndices[i]]
                preparedInput, numPadded = self.prepareInput(instance)
                preparedInput = preparedInput[None, :, :]
                dataSeq = self.getSequence(instance)
                pathToInstance = instance["instanceFolderPath"]
                if trainHistoryPath and f"{pathToInstance}_{epoch}" in trainHistory.values:
                    continue

                goldLabels, goldLabelsChar, labelTranslation, attentionMask = self.getGoldLabels(dataSeq, instance,
                                                                                                 numPadded)
                goldLabels = torch.tensor(goldLabels)

                self.zero_grad()
                loss, predictions = self.forward(preparedInput, goldLabels, attentionMask)
                batchData = batchData.append(
                    {'epoch': epoch + 1, 'batch': i + 1, 'loss': loss.item()},
                    ignore_index=True)
                loss.backward()
                optimizer.step()
                overallEpochLoss += loss.item()

                if trainHistoryPath:
                    trainHistory.loc[len(trainHistory)] = f"{pathToInstance}_{epoch}"
            overallEpochLoss = overallEpochLoss / len(dataset)
            print(f"Avg. loss for epoch {epoch + 1}: {overallEpochLoss}")
            epochData = epochData.append({'epoch': epoch + 1, 'avg_loss': overallEpochLoss}, ignore_index=True)

        time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        epochData.to_csv(f"./trainEpochData_{time}.csv")
        batchData.to_csv(f"./trainBatchData_{time}.csv")

        if trainHistoryPath:
            trainHistory.to_csv(trainHistoryPath)

        print("Training of BERT-CRF complete")


def testModel(self, dataset):
    testResults = pd.DataFrame(columns=['invoiceInstance', 'prediction', "goldLabels", "instanceLoss"])
    self.eval()

    with torch.no_grad:
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

    print("Testing of BERT-CRF complete")
    return testResults


if __name__ == '__main__':
    data = CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
    invoice_BiLSTM_CNN_CRF = Invoice_BiLSTM_CNN_CRF(data)
    # temp = invoice_BiLSTM_CNN_CRF.prepareInput(data[0])
    # invoice_BiLSTM_CNN_CRF.trainModel(data)
    invoice_BiLSTM_CNN_CRF.trainModel(data)
