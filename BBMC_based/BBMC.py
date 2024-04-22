import string
from transformers import BertTokenizerFast, BertModel
import torch
# from TorchCRF import CRF
from torchcrf import CRF
import pandas as pd
from utils.helperFunctions import loadJSON, getConfig, CONFIG_PATH, separate
from datetime import datetime
from dataProcessing.customDataset import CustomDataset
import warnings

# Suppress the pandas append deprecation warning
warnings.filterwarnings(
    action='ignore',
    message="The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead."
)
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


class InvoiceBBMC(torch.nn.Module):

    def __init__(self, tokenizer=tokenizer, model=model, hiddenDim=300, LSTMlayers=100, dropout_rate=.5,
                 outputDim=19):
        super(InvoiceBBMC, self).__init__()

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

    def forward(self, inputTensor, labels=None):

        outputs = self.bert(inputTensor["input_ids"], attention_mask=inputTensor["attention_mask"])
        x = outputs[0]

        x, _ = self.bilstm(x)

        x, _ = self.multiheadAttention(x, x, x)

        emissions = self.fc1(x)

        # Get rid of CLS tokens for the CRF
        emissions = emissions[:, 1:-1]

        mask = [1 for i in range(emissions.size(1))]
        mask = torch.tensor([mask]).bool()

        if labels is not None:
            loss = -self.crf(emissions, labels, reduction='mean', mask=mask)
            tags = self.crf.decode(emissions)
            return loss, tags
        else:
            prediction = self.crf.decode(emissions, mask=mask)
            return -1, prediction

    def getGoldLabels(self, dataInstance, tokenizedSequence):

        dataSequenceAsList = self.getSequence(dataInstance).split(" ")
        dataSequence = "".join(dataSequenceAsList).lower()
        groundTruth = dataInstance["goldLabels"]

        goldLabelsChar = ["O" for i in range(tokenizedSequence["input_ids"].size(1) - 2)]
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
                                                       tokenizedSequence.encodings[0].offsets)
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

    def trainModel(self, numEpochs, dataset, lr=1e-3):

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)

        epochData = pd.DataFrame(columns=['epoch', 'avgLoss'])
        batchData = pd.DataFrame(columns=['epoch', 'batch', 'loss'])

        for epoch in range(numEpochs):
            print(f"Epoch {epoch + 1} / {numEpochs}")

            overallEpochLoss = 0
            shuffledIndices = torch.randperm(len(dataset))
            for i in range(len(dataset)):
                instance = dataset[shuffledIndices[i]]
                preparedInput = self.prepareInput(instance)

                # self.getGoldLabels2(instance, preparedInput)
                goldLabels, goldLabelsChar, labelTranslation = self.getGoldLabels(instance, preparedInput)

                goldLabels = torch.tensor([goldLabels])

                self.zero_grad()
                loss, predictions = self.forward(preparedInput, goldLabels)
                batchData = batchData.append(
                    {'epoch': epoch + 1, 'batch': i + 1, 'loss': loss.item()},
                    ignore_index=True)

                loss.backward()
                optimizer.step()
                overallEpochLoss += loss.item()

            overallEpochLoss = overallEpochLoss / len(dataset)
            epochData = epochData.append({'epoch': epoch + 1, 'avg_loss': overallEpochLoss}, ignore_index=True)
            print(f"Avg. loss for epoch {epoch + 1}: {overallEpochLoss}")

        time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        epochData.to_csv(f"./trainEpochData_{time}.csv")
        batchData.to_csv(f"./trainBatchData_{time}.csv")
        print("Training of BBMC model complete")

    def testModel(self, dataset):
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


if __name__ == '__main__':
    data = CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
    invoiceBBMC = InvoiceBBMC()

    torch.save(invoiceBBMC.state_dict(), getConfig("BBMC_based", CONFIG_PATH)["pathToStateDict"])
    invoiceBBMC.load_state_dict(torch.load(getConfig("BBMC_based", CONFIG_PATH)["pathToStateDict"]))
    # invoiceBBMC.trainModel(2, data)
    invoiceBBMC.testModel(data)
