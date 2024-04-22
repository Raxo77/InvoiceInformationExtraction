import string

import torch
import torchtext
#from TorchCRF import CRF
from torchcrf import CRF
import pandas as pd
from collections import OrderedDict
from utils.helperFunctions import getConfig, CONFIG_PATH
from dataProcessing.customDataset import CustomDataset
import nltk
from nltk.util import ngrams

"""
For embedding of words: Test with FastText and GloVe and check which has better performance
"""

CONFIG_PATH = CONFIG_PATH


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
        self.fc1 = torch.nn.Linear(512, 50)
        self.crf = CRF(10, batch_first=True)

    def getSequence(self, dataInstance):

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

    def getGoldLabels(self, numPadded):
        pass

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
                preparedInput, numPadded = self.prepareInput(instance)
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
    # temp = invoice_BiLSTM_CNN_CRF.prepareInput(data[0])
    invoice_BiLSTM_CNN_CRF.trainModel(data)
