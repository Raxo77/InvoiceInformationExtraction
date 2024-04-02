import torch
import torchtext
from TorchCRF import CRF
import pandas as pd
from collections import OrderedDict
from utils.CONFIG_PATH import CONFIG_PATH
from utils.helperFunctions import getConfig
from dataProcessing.customDataset import CustomDataset

"""
For embedding of words: Test with FastText and GloVe and check which has better performance
"""


def getPrintableChars():
    # based on ISO/IEC 8859-1
    printableChars = []
    for i in range(32, 127):
        printableChars.append(chr(i))
    for i in range(160, 256):
        printableChars.append(chr(i))

    return printableChars


def createVocab(charList):
    vocab = torchtext.vocab.vocab(OrderedDict([(char, 1) for char in charList]), specials=["<unk>", "<eos>"])
    return vocab


"""
For the max length of words 40 is taken as initial estimate
"""


class Invoice_BiLSTM_CNN_CRF(torch.nn.Module):

    def __init__(self, dataset, wordEmbedding="fastText", maxWordsPerInvoice=512, charEmbeddingSize=30, kernelSize=3):
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
        if wordEmbedding == "fastText":
            weight = None
            wordEmbeddingSize = None
        elif wordEmbedding == "GloVe":
            weight = None
            wordEmbeddingSize = None
        else:
            print("Invalid entry for ´wordEmbedding´ --> defaulting to fastText")
            weight = None
            wordEmbeddingSize = None

        self.wordEmbedding = torch.nn.Embedding.from_pretrained(weight)

        # BiLSTM-CRF
        self.bilstm = torch.nn.LSTM(wordEmbeddingSize + charEmbeddingSize, 512 // 2, bidirectional=True)
        self.fc1 = torch.nn.Linear(512, 50)
        self.crf = CRF(10, batch_first=True)

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
        pass

    def prepareInput(self, dataInstance):

        invoiceSeq = self.getSequence(dataInstance)

        charEmbeddings = self.embedChars(invoiceSeq)
        charEmbeddings = torch.permute(charEmbeddings, (0, 2, 1))
        charEmbeddings = self.conv1d(charEmbeddings)
        charEmbeddings = torch.permute(charEmbeddings, (0, 2, 1))
        charEmbeddings, _ = self.charLSTM(charEmbeddings)
        charEmbeddings = charEmbeddings[:, -1, :]

        wordEmbeddings = self.embedWords()

        preparedInput = torch.cat(wordEmbeddings, charEmbeddings)
        return preparedInput

    def forward(self, inputTensor, labels=None):

        x, _ = self.bilstm(inputTensor)
        emissions = self.fc1(x)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=inputTensor["attention_mask"].byte(), reduction='mean')
            tags = self.crf.decode(emissions, mask=inputTensor["attention_mask"].byte())
            return loss, tags
        else:
            prediction = self.crf.decode(emissions, mask=inputTensor["attention_mask"].byte())
            return -1, prediction

    def trainModel(self, dataset, numEpochs=40, lr=1e-3):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)

        self.train()
        for epoch in range(numEpochs):
            print(f"Epoch {epoch + 1} / {numEpochs}")
            lossPerBatch = []
            shuffledIndices = torch.randperm(len(dataset))
            for i in range(len(dataset)):
                instance = dataset[shuffledIndices[i]]
                preparedInput = self.prepareInput(instance)
                goldLabels = 0  # TODO !!!
                self.zero_grad()
                loss, predictions = self.forward(preparedInput, goldLabels)
                loss.backward()
                optimizer.step()
                print(i, loss)
                lossPerBatch.append(loss.item())
            print(f"Epoch: {epoch + 1}/{numEpochs}")


if __name__ == '__main__':
    data = CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
    invoice_BiLSTM_CNN_CRF = Invoice_BiLSTM_CNN_CRF(data)
    temp = invoice_BiLSTM_CNN_CRF.prepareInput(data[0])

    print(temp)
    print(temp.size())
