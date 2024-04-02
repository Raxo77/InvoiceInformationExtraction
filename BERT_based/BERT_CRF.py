import string

from transformers import BertTokenizer, BertModel
import torch
import os
import pandas as pd
from TorchCRF import CRF
from utils.CONFIG_PATH import CONFIG_PATH
from utils.helperFunctions import loadJSON, getConfig
from dataProcessing.customDataset import CustomDataset

TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased')
MODEL = BertModel.from_pretrained('bert-base-cased')
tokenizer = TOKENIZER
model = MODEL


def flattenList(nestedList):
    flatList = []
    for element in nestedList:
        if isinstance(element, list):
            flatList.extend(flattenList(element))
        else:
            flatList.append(element)
    return flatList


def check_sequence_existence(tokens_text, tokens_seq):
    for i in range(len(tokens_text) - len(tokens_seq) + 1):
        l1 = tokens_text[i:i + len(tokens_seq)]
        l1 = [j.lower() for j in l1]
        l2 = [j.lower() for j in tokens_seq]
        # print(l1,l2)
        if l1 == l2:
            return True, i  # Returns True and start index if sequence is found
    return False, -1  # Returns False and -1 if sequence is not found


def getGoldData(path: str):
    goldData = loadJSON(path)
    temp = goldData
    goldData = {}
    for key, value in temp.items():
        if value is not None:
            goldData[key] = tokenizer.tokenize(value["value"])
        else:
            goldData[key] = None
    return goldData


class InvoiceBERT(torch.nn.Module):

    def __init__(self, dataset, tokenizer=TOKENIZER, model=MODEL, featureSize=2316, numLabels=9):
        super(InvoiceBERT, self).__init__()

        self.hiddenSize = model.config.hidden_size
        self.inputProjection = torch.nn.Linear(featureSize, self.hiddenSize)

        self.tokenizer = tokenizer
        self.embeddingLayer = model.embeddings
        self.BERTencoders = model.encoder
        self.BERTpooler = model.pooler

        self.classifier = torch.nn.Linear(self.hiddenSize, numLabels)
        self.crf = CRF(numLabels, batch_first=True)

        self.dataset = dataset
        if len(dataset) < 1:
            print("Warning: Provided Dataset is empty")

        # For testing and illustratory purposes only
        self.sampleSequence = self.getSequence(self.dataset.__getitem__(0))
        self.sampleTokens, self.sampleTokenIDdict = self.tokenizeSequence(self.sampleSequence)
        self.sampleEmbeddings, self.sampleEmbeddingsDict = self.embedSequence(self.sampleTokens,
                                                                              self.sampleTokenIDdict["input_ids"])

    def getSequence(self, dataInstance):
        featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
        colNames = list(featuresDF.columns)
        colNames[0] = "wordKey"
        featuresDF.columns = colNames

        seqString = ""
        seqList = list(map(lambda x: x.split("_")[0], featuresDF["wordKey"]))

        seqString += seqList[0]

        for i in seqList[1:]:
            if i in string.punctuation:
                seqString += i
            else:
                seqString += f" {i}"

        return seqString

    def tokenizeSequence(self, inputSequence):

        tokenIDdict = self.tokenizer(inputSequence)
        tokens = self.tokenizer.convert_ids_to_tokens(tokenIDdict["input_ids"])
        tokenIDdict = self.tokenizer(inputSequence, return_tensors="pt")

        return tokens, tokenIDdict

    def embedSequence(self, tokens, tokenIDtensor):
        embeddings = self.embeddingLayer(tokenIDtensor)

        embeddingsDict = {}
        for count, temp in enumerate(zip(tokens, embeddings[0])):
            token, embeddingVector = temp

            embeddingsDict[f"{token}_{count}"] = embeddingVector

        return embeddings, embeddingsDict

    def getSurroundingWordEmbeddings(self, featuresDF, instanceSeq, instanceEmbeddingsDict):

        surroundingWords = featuresDF.loc[:,
                           ["topmost", "bottommost", "left", "right", "above", "below"]].values.tolist()
        surroundingWords = [[str(j) for j in i if not str(j).startswith("nan")] for i in surroundingWords]
        surroundingWords = {featuresDF.loc[count, "wordKey"]: i for count, i in enumerate(surroundingWords)}

        sequenceWords = instanceSeq.split(" ")
        invoiceWords = featuresDF["wordKey"].tolist()
        matchedList = []

        for count, seqWord in enumerate(sequenceWords):
            for invoiceWord in invoiceWords:
                if invoiceWord.split("_")[0] in seqWord.split("_")[0]:
                    matchedList.append([f"{seqWord}_{count}", invoiceWord])
                    invoiceWords = invoiceWords[1:]
                else:
                    break

        count = 1
        for i in matchedList:
            temp = []
            for token in tokenizer.tokenize(i[1].split("_")[0]):
                temp.append(f"{token}_{count}")
                count += 1
            i.append(temp)

        for index, i in enumerate(matchedList):
            i.append(surroundingWords[i[1]])

        matchedListDF = pd.DataFrame(matchedList)

        for surrWordsList in matchedList:
            temp = []
            for surrWord in surrWordsList[3]:
                temp.append(matchedListDF.loc[matchedListDF[1] == surrWord, 2].tolist())
            surrWordsList.append(flattenList(temp))

        for surrWordsList in matchedList:
            temp = []
            for surrWord in surrWordsList[3]:
                temp.append([i for i in pd.DataFrame(matchedList)[1].tolist() if i == surrWord])

        for bigList in matchedList:
            allEmbs = []
            for token in flattenList(bigList[4]):
                allEmbs.append(instanceEmbeddingsDict[token])
            bigList.append(torch.mean(torch.stack(allEmbs), dim=0))

        someDFi = pd.DataFrame(matchedList).iloc[:, [2, 5]].explode(2)
        someListi = someDFi.loc[:, 5].tolist()

        # Add "CLS" and "PUNCT" tokens
        someListi.insert(0, torch.zeros(self.hiddenSize))
        someListi.append(torch.zeros(self.hiddenSize))

        someTensori = torch.stack(someListi, dim=0)
        someTensori = someTensori.reshape(1, someTensori.size()[0], self.hiddenSize)

        return someTensori

    def embedPatternFeatures(self, featuresDF, instanceSeq):
        surrWords = featuresDF.loc[:, ["standardisedText"]].values.tolist()
        surrWords = [[str(j) for j in i if not str(j).startswith("nan")] for i in surrWords]
        surrWords = {featuresDF.loc[count, "wordKey"]: i for count, i in enumerate(surrWords)}

        sequenceWords = [i for i in instanceSeq.split(" ")]
        invoiceWords = featuresDF["wordKey"].tolist()
        matchedList = []

        for count, seqWord in enumerate(sequenceWords):
            for invoiceWord in invoiceWords:
                if invoiceWord.split("_")[0] in seqWord.split("_")[0]:
                    matchedList.append([f"{seqWord}_{count}", invoiceWord])
                    invoiceWords = invoiceWords[1:]
                else:
                    break

        count = 1
        for index, i in enumerate(matchedList):
            temp = []
            for token in tokenizer.tokenize(i[1].split("_")[0]):
                temp.append(f"{token}_{count}")
                count += 1
                # print(count)
            i.append(temp)

        for index, i in enumerate(matchedList):
            i.append(surrWords[i[1]])

        df = pd.DataFrame(matchedList)

        pattern_features = df.explode(2)[3].apply(lambda x: "".join(x)).tolist()
        toks = self.tokenizer(pattern_features, padding=True, truncation=True, return_tensors="pt")
        embs = self.embeddingLayer(toks["input_ids"])

        embs = torch.sum(embs, dim=1)
        torch.transpose(embs, 0, 1)
        embs = torch.cat((embs, torch.zeros(1, 768)), dim=0)
        embs = torch.cat((torch.zeros(1, 768), embs), dim=0)
        embs = embs.reshape(1, embs.size()[0], embs.size()[1])

        return embs, df

    def normaliseNumericFeatures(self, featuresDF, patternFeaturesReturn):
        numericData = featuresDF.select_dtypes(include=["number"])
        numericData = (numericData - numericData.mean()) / numericData.std()
        numericData["wordKey"] = featuresDF["wordKey"]
        numericData["tokens"] = patternFeaturesReturn[1][2]
        numericData = numericData.explode("tokens")
        numericData = numericData.select_dtypes(include=["number"])
        numericData = torch.tensor(numericData.values)
        numericData = torch.cat((numericData, torch.zeros(1, 7)), dim=0)
        numericData = torch.cat((torch.zeros(1, 7), numericData), dim=0)
        numericData = numericData.reshape(1, numericData.size()[0], numericData.size()[1])
        return numericData

    def oneHotCategorical(self, featureDF, patternFeaturesReturn):
        categoricalData = featureDF.select_dtypes(include=["bool"])
        categoricalData = pd.get_dummies(data=categoricalData).astype(int)
        categoricalData["wordKey"] = featureDF["wordKey"]
        categoricalData["tokens"] = patternFeaturesReturn[1][2]
        categoricalData = categoricalData.explode("tokens")
        categoricalData = categoricalData.select_dtypes(include=["int"])
        categoricalData = torch.tensor(categoricalData.values)

        categoricalData = torch.cat((categoricalData, torch.zeros(1, 5)), dim=0)
        categoricalData = torch.cat((torch.zeros(1, 5), categoricalData), dim=0)
        categoricalData = categoricalData.reshape(1, categoricalData.size()[0], categoricalData.size()[1])

        return categoricalData

    def prepareInput(self, dataInstance):

        featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
        colNames = list(featuresDF.columns)
        colNames[0] = "wordKey"
        featuresDF.columns = colNames

        instanceSequence = self.getSequence(dataInstance)
        instanceTokens, instanceTokenIDdict = self.tokenizeSequence(instanceSequence)
        instanceEmbeddings, instanceEmbeddingsDict = self.embedSequence(instanceTokens,
                                                                        instanceTokenIDdict["input_ids"])

        mainEmbeddings = instanceEmbeddings

        surroundingEmbeddings = self.getSurroundingWordEmbeddings(featuresDF,
                                                                  instanceSequence,
                                                                  instanceEmbeddingsDict
                                                                  )
        embs, df = self.embedPatternFeatures(featuresDF,
                                             instanceSequence
                                             )

        normalisedNumerics = self.normaliseNumericFeatures(featuresDF,
                                                           (embs, df)
                                                           )

        oneHotCategorical = self.oneHotCategorical(featuresDF,
                                                   (embs, df)
                                                   )

        torchInput = torch.concat(
            (mainEmbeddings,
             surroundingEmbeddings,
             embs,
             normalisedNumerics,
             oneHotCategorical),
            dim=2
        )

        return torchInput

    def labelTokens(self, seqTokens, goldData):
        tokenLabels = [0 for i in seqTokens]
        for count, i in enumerate(goldData.values()):
            if i is not None:
                temp = check_sequence_existence(seqTokens, i)
                if temp[0]:
                    for tha in range(temp[1], temp[1] + len(i)):
                        tokenLabels[tha] = count + 1
        return torch.tensor([tokenLabels])

    def forward(self, inputTensor, labels=None):

        projectedInput = self.inputProjection(inputTensor)

        outputsAll = self.BERTencoders(projectedInput)
        outputs = outputsAll.last_hidden_state
        emissions = self.classifier(outputs)

        mask = torch.ones(outputs.size()[1])
        mask = mask.reshape(1, -1)

        #
        if labels is not None:
            loss = -self.crf.forward(emissions, labels, mask=mask.byte(), reduction="mean")
            tags = self.crf.decode(emissions, mask=mask.byte())
            return loss, tags
        else:
            return self.crf.decode(emissions, mask=mask.byte())

    def trainModel(self, numEpochs, dataset, lr=1e-3):

        resList = []

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)

        self.train()
        for epoch in range(numEpochs):
            print(f"Epoch {epoch + 1} / {numEpochs}")
            overallLoss = 0
            shuffledIndices = torch.randperm(len(dataset))
            for i in range(len(dataset)):
                # get one specific invoice
                # That is to say, currently invoices are fed in batches of 1 to the algorithm
                dataInstance = dataset[shuffledIndices[i]]
                preparedInput = self.prepareInput(dataInstance)
                preparedInput = preparedInput.type(dtype=torch.float32)

                instanceSequence = self.getSequence(dataInstance)
                instanceTokens, instanceTokensDict = self.tokenizeSequence(instanceSequence)
                labels = self.labelTokens(instanceTokens,
                                          getGoldData(
                                              os.path.join(dataInstance["instanceFolderPath"], "goldLabels.json"))
                                          )

                self.zero_grad()
                loss, tags = self.forward(preparedInput, labels)
                resList.append((loss, tags))
                loss.backward()
                optimizer.step()
                overallLoss += loss.item()
            print(overallLoss)
        return resList


if __name__ == "__main__":
    # def train(model, numEpochs, trainingData, lr=1e-4):
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)
    #
    #     trainDataLoader = DataLoader(trainingData, batch_size=1, shuffle=True)
    #
    #     model.train()
    #     initial_sums = ([torch.sum(p.data) for p in model.parameters() if p.requires_grad]
    #     )
    #     for epoch in range(numEpochs):
    #         overallLoss = 0
    #         for i in range(1):
    #             inputTensor = model.prepareInput().type(dtype=torch.float32)
    #             labels = labelTokens(model.tokens, getGoldData(r"C:\Users\fabia\NER_for_IIE\data\00003\goldLabels.json"))
    #
    #             model.zero_grad()
    #
    #             loss, tags = model.forward(inputTensor, labels)
    #             loss.backward()
    #             optimizer.step()
    #
    #             overallLoss += loss.item()
    #     post_training_sums = ([torch.sum(p.data) for p in model.parameters() if p.requires_grad])
    #     weights_changed = any(initial != post for initial, post in zip(initial_sums, post_training_sums))
    #     print("Weights changed:", weights_changed)

    data = CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
    invoiceBERT = InvoiceBERT(data)
    initial_sums = [torch.sum(p.data) for p in invoiceBERT.parameters() if p.requires_grad]
    resList = invoiceBERT.trainModel(100, data)
    post_training_sums = [torch.sum(p.data) for p in invoiceBERT.parameters() if p.requires_grad]
    weights_changed = any(initial != post for initial, post in zip(initial_sums, post_training_sums))
    print("Weights changed:", weights_changed)
    print(resList[0])
    print(resList[len(resList) // 2])
    print(resList[-1])
