from transformers import BertTokenizer, BertModel
import torch
from TorchCRF import CRF
import pandas as pd
from utils.helperFunctions import loadJSON, getConfig, CONFIG_PATH
from dataProcessing.customDataset import CustomDataset


TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased')
MODEL = BertModel.from_pretrained('bert-base-cased')
tokenizer = TOKENIZER
model = MODEL


class InvoiceBBMC(torch.nn.Module):

    def __init__(self, dataset, tokenizer=tokenizer, model=model, hiddenDim=300, LSTMlayers=100, dropout_rate=.5,
                 outputDim=10):
        super(InvoiceBBMC, self).__init__()

        self.data = dataset

        self.tokenizer = tokenizer
        self.bert = model

        embeddingDim = self.bert.config.to_dict()['hidden_size']
        self.bilstm = torch.nn.LSTM(embeddingDim, hiddenDim // 2, num_layers=LSTMlayers, bidirectional=True,
                                    batch_first=True)

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.multiheadAttention = torch.nn.MultiheadAttention(hiddenDim, num_heads=3, batch_first=True)
        self.fc1 = torch.nn.Linear(hiddenDim, outputDim)
        self.crf = CRF(outputDim, batch_first=True)

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

    def prepareInput(self, dataInstance):

        inputSeq = self.getSequence(dataInstance)
        tokenizedSeq = self.tokenizer(inputSeq, return_tensors="pt")

        return tokenizedSeq

    def forward(self, inputTensor, labels=None):

        outputs = self.bert(inputTensor["input_ids"], attention_mask=inputTensor["attention_mask"])
        x = outputs[0]

        x, _ = self.bilstm(x)

        x, _ = self.multiheadAttention(x, x, x)

        emissions = self.fc1(x)

        if labels is not None:
            loss = -self.crf(emissions, labels, mask=inputTensor["attention_mask"].byte(), reduction='mean')
            tags = self.crf.decode(emissions, mask=inputTensor["attention_mask"].byte())
            return loss, tags
        else:
            prediction = self.crf.decode(emissions, mask=inputTensor["attention_mask"].byte())
            return -1, prediction

    def trainModel(self, numEpochs, dataset, lr=1e-3):

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
                goldLabels = torch.randn(1, preparedInput["input_ids"].size()[-1])
                goldLabels = goldLabels.type(torch.int32)
                self.zero_grad()
                loss, predictions = self.forward(preparedInput, goldLabels)
                loss.backward()
                optimizer.step()
                print(i, loss)
                lossPerBatch.append(loss.item())
            print(f"Epoch: {epoch + 1}/{numEpochs}")


if __name__ == '__main__':
    data = CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
    invoiceBBMC = InvoiceBBMC(data)
    invoiceBBMC.trainModel(10, data)
