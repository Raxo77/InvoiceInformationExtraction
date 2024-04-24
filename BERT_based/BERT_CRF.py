import os
import torch
import string
import warnings
import pandas as pd
from torchcrf import CRF
# from TorchCRF import CRF
from datetime import datetime
from transformers import BertTokenizerFast, BertModel
from dataProcessing.customDataset import CustomDataset
from utils.helperFunctions import loadJSON, getConfig, CONFIG_PATH, flattenList

torch.manual_seed(123)

warnings.filterwarnings("ignore", message="where received a uint8 condition tensor.*")
warnings.filterwarnings(action='ignore', message="The frame.append method is deprecated and will be removed from ")

TOKENIZER = BertTokenizerFast.from_pretrained('bert-base-cased')
MODEL = BertModel.from_pretrained('bert-base-cased')
tokenizer = TOKENIZER
model = MODEL


def generate_ngrams(word, n):
    """ Generate n-grams from a given word or return the word itself if n is greater than the word length. """
    if len(word) >= n:
        return [word[i:i + n] for i in range(len(word) - n + 1)]
    else:
        return [word]  # Return the word as a single n-gram if it's shorter than the specified n-gram size


def ngrams_to_indices(ngrams, vocab):
    """ Convert n-grams to indices based on a vocabulary dictionary """
    return [vocab[ng] for ng in ngrams if ng in vocab]


def words_to_ngrams(words, ngram_sizes):
    """ Convert words to a list of n-grams based on their respective sizes """
    all_ngrams = [generate_ngrams(word, size) for word, size in zip(words, ngram_sizes)]
    return all_ngrams


def getGoldData(path: str) -> dict:
    goldData = loadJSON(path)
    temp = goldData
    goldData = {}
    for key, value in temp.items():
        if value is not None:
            goldData[key] = tokenizer.tokenize(value["value"])
        else:
            goldData[key] = None
    return goldData


def countTokensPerWord2(wordSeq: str, offsets: list) -> list:
    wordIndex = 0
    tokenCount = [0 for i in range(len(wordSeq.split(" ")))]
    for count in range(len(offsets) - 1):
        tokenCount[wordIndex] += 1
        if offsets[count][1] != offsets[count + 1][0]:
            wordIndex += 1
    # with the increment-before-if-approach, the first one is counted one time too many
    tokenCount[0] -= 1
    tokenCount[-1] = len(offsets) - sum(tokenCount)
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


def checkSequenceExistence(tokens_text, tokens_seq):
    for i in range(len(tokens_text) - len(tokens_seq) + 1):
        l1 = tokens_text[i:i + len(tokens_seq)]
        l1 = [j.lower() for j in l1]
        l2 = [j.lower() for j in tokens_seq]
        # print(l1,l2)
        if l1 == l2:
            return True, i  # Returns True and start index if sequence is found
    return False, -1  # Returns False and -1 if sequence is not found


class InvoiceBERT(torch.nn.Module):

    def __init__(self,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 tokenizer=TOKENIZER,
                 model=MODEL,
                 featureSize=2316,
                 numLabels=10
                 ):
        super(InvoiceBERT, self).__init__()

        self.device = device
        self.featureSize = featureSize
        self.hiddenSize = model.config.hidden_size

        self.inputProjection = torch.nn.Linear(featureSize, self.hiddenSize).to(device)

        self.tokenizer = tokenizer
        self.embeddingLayer = model.embeddings.to(device)

        self.BERTencoders = model.encoder.to(device)
        self.BERTpooler = model.pooler.to(device)

        self.classifier = torch.nn.Linear(self.hiddenSize, numLabels).to(device)
        self.crf = CRF(numLabels, batch_first=True).to(device)

    def getSequence(self, dataInstance):

        featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
        colNames = list(featuresDF.columns)
        colNames[0] = "wordKey"
        featuresDF.columns = colNames

        seqString = ""
        seqList = list(map(lambda x: x.split("_")[0], featuresDF["wordKey"]))

        # first word in sequence (no leading whitespace)
        seqString += seqList[0]

        for i in seqList[1:]:
            if i in ['.', ',', '?', '!']:
                seqString += i
            else:
                seqString += f" {i}"

        return seqString

    def tokenizeSequence(self, inputSequence):
        """
        :param inputSequence: string of the hOCR output
        :return: a tuple of the results of the natural language tokens and the output of the tokenizer for the sequence
        """

        tokenIDdict = self.tokenizer(inputSequence)
        tokens = self.tokenizer.convert_ids_to_tokens(tokenIDdict["input_ids"])
        tokenIDdict = self.tokenizer(inputSequence, return_tensors="pt")

        return tokens, tokenIDdict

    def embedSequence(self, tokens, tokenIDtensor):
        """

        :param tokens: list of the natural language version of the tokens
        :param tokenIDtensor: tensor containing the token IDs (i.e., the output of the tokenizer)
        :return: tuple of the tensor of the token embeddings and a dict mapping tokens to their respective embedding
        """
        tokenIDtensor = tokenIDtensor.to(self.device)
        # embedding layer returns tensor of (batchSize, numTokens, embeddingDim)
        embeddings = self.embeddingLayer(tokenIDtensor)

        embeddingsDict = {}
        for count, temp in enumerate(zip(tokens, embeddings[0])):
            token, embeddingVector = temp

            embeddingsDict[f"{token}_{count}"] = embeddingVector

        return embeddings, embeddingsDict

    def getSurroundingWordEmbeddings(self, featuresDF, tokenDF, instanceSeq, instanceEmbeddingsDict):

        # Create a dict with words in the hOCR sequence as keys and a list of the corresponding surrounding words as values
        surroundingWords = featuresDF.loc[:, ["wordKey", "topmost", "bottommost", "left", "right", "above", "below"]]
        surrWordEmbeddings = []

        surrWordsPerToken = pd.merge(tokenDF, featuresDF, left_on=1, right_on='wordKey', how='left').explode(
            2).reset_index()
        for row in range(len(surrWordsPerToken)):
            selectedRow = surrWordsPerToken.loc[
                row, ["topmost", "bottommost", "left", "right", "above", "below"]]
            avgEmbeds = list(map(lambda x: str(x).split("_")[0], selectedRow.tolist()))
            avgEmbeds = self.tokenizer(avgEmbeds, add_special_tokens=False)
            tempEmbeds = []
            for i in avgEmbeds["input_ids"]:
                if len(i) > 0:
                    tempEmbeds.append(self.embeddingLayer(torch.tensor([i], device=self.device)).mean(dim=1))
            surrWordEmbeddings.append(torch.stack(tempEmbeds).mean(dim=0))

        surrWordEmbeddings = torch.stack(surrWordEmbeddings)
        surrWordEmbeddings = surrWordEmbeddings.transpose(0, 1)
        surrWordEmbeddings = torch.cat(
            (self.embeddingLayer(torch.tensor([[101]], device=self.device)), surrWordEmbeddings),
            dim=1)
        surrWordEmbeddings = torch.cat(
            (surrWordEmbeddings, self.embeddingLayer(torch.tensor([[102]], device=self.device))),
            dim=1)

        return surrWordEmbeddings

    def embedPatternFeatures(self, featuresDF, instanceSeq):
        # featuresDF in this context also contains punctuation

        # Given the unnatural structure of the pattern features, tokenization does not work properly for those features
        # Consequently, the number of tokens per word in the original sequence is counted and each pattern feature
        # is split into ngrams where n corresponds to the number of tokens per word. The ngrams are then used to
        # get the indices for the embedding and then use these embeddings.
        # While the embedding of a pattern ngram may still not be optimal, this approach allows for a level of
        # tokenization of the pattern features -- in this sense, rather closely resembles char embedding
        patternDF = featuresDF.loc[:, ["wordKey", "standardisedText"]]

        tokensOriginalSeq = self.tokenizer(instanceSeq, return_offsets_mapping=True)
        tokensPerWordNaturalLanguage = countTokensPerWord(instanceSeq, tokensOriginalSeq.encodings[0].offsets[1:-1])

        patternList = []
        for index, tokenCount in enumerate(tokensPerWordNaturalLanguage):
            temp = tokenCount
            while temp != 0:
                patternList.append(patternDF.iloc[index, 1])
                temp -= 1

        patternEmbeds = []
        for i in patternList:
            temp = tokenizer(i, return_tensors="pt")
            tempEmbeds = self.embeddingLayer(temp["input_ids"].to(self.device)).mean(dim=1)
            patternEmbeds.append(tempEmbeds)

        patternEmbeds = torch.stack(patternEmbeds)
        patternEmbeds = patternEmbeds.view(-1, 768)
        # patternEmbeds = torch.cat((patternEmbeds, torch.zeros(1, 768, device=self.device)), dim=0)
        # patternEmbeds = torch.cat((torch.zeros(1, 768, device=self.device), patternEmbeds), dim=0)

        surrWords = featuresDF.loc[:, ["standardisedText"]].values.tolist()
        surrWords = [[str(j) for j in i if not str(j).startswith("nan")] for i in surrWords]
        surrWords = {featuresDF.loc[count, "wordKey"]: i for count, i in enumerate(surrWords)}

        df = pd.DataFrame({0: [], 1: [], 2: [], 3: []})
        matchedList = []
        textTokens = self.tokenizer.convert_ids_to_tokens(tokensOriginalSeq["input_ids"])[1:-1]

        position = 0
        for index, tokenCount in enumerate(tokensPerWordNaturalLanguage):
            tempList = []
            for i in range(tokenCount):
                tempList.append(f"{textTokens[position]}_{position + 1}")
                position += 1
            matchedList.append(tempList)

        df[2] = matchedList
        df[1] = patternDF["wordKey"]
        patternEmbeds = patternEmbeds.view(1, -1, 768)
        patternEmbeds = torch.cat((self.embeddingLayer(torch.tensor([[101]], device=self.device)), patternEmbeds),
                                  dim=1)
        patternEmbeds = torch.cat((patternEmbeds, self.embeddingLayer(torch.tensor([[102]], device=self.device))),
                                  dim=1)

        return patternEmbeds, df

    def normaliseNumericFeatures(self, featuresDF, patternFeaturesReturn):
        tokenDF = patternFeaturesReturn[1]
        # Hard-coded colNames necessary/helpful to catch corner case invoices (cf. 04424)
        # numericData = featuresDF.select_dtypes(include=["number"])
        numericData = featuresDF.loc[:,
                      ['normTop', 'normLeft', 'normBottom', 'normRight', 'wordWidth', 'wordHeight', 'wordArea']]
        # normalise numeric data
        numericData = (numericData - numericData.mean()) / numericData.std()

        # numeric features are calculated for whole words, invoiceBERT uses subwords for training
        # --> explode df to match word features to token features
        numericData["wordKey"] = featuresDF["wordKey"]

        numericData = pd.merge(tokenDF, featuresDF, left_on=1, right_on='wordKey', how='left').explode(2).reset_index()

        numericData = numericData.loc[:,
                      ['normTop', 'normLeft', 'normBottom', 'normRight', 'wordWidth', 'wordHeight', 'wordArea']]
        numericData = torch.tensor(numericData.values)

        numericData = torch.cat((numericData, torch.zeros(1, 7)), dim=0)
        numericData = torch.cat((torch.zeros(1, 7), numericData), dim=0)

        numericData = numericData.reshape(1, numericData.size(0), numericData.size(1))
        return numericData

    def oneHotCategorical(self, featureDF, patternFeaturesReturn):
        tokenDF = patternFeaturesReturn[1]
        categoricalData = featureDF.select_dtypes(include=["bool"])
        # ensure boolean variables are encoded as integers
        categoricalData = pd.get_dummies(data=categoricalData).astype(int)
        categoricalData["wordKey"] = featureDF["wordKey"]

        categoricalData = pd.merge(tokenDF, featureDF, left_on=1, right_on='wordKey', how='left').explode(
            2).reset_index()

        categoricalData = categoricalData.select_dtypes(include=["bool"])
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

        embs, df = self.embedPatternFeatures(featuresDF, instanceSequence)
        surroundingEmbeddings = self.getSurroundingWordEmbeddings(featuresDF, df, instanceSequence,
                                                                  instanceEmbeddingsDict)

        normalisedNumerics = self.normaliseNumericFeatures(featuresDF, (embs, df))
        normalisedNumerics = normalisedNumerics.to(self.device)

        oneHotCategorical = self.oneHotCategorical(featuresDF, (embs, df))
        oneHotCategorical = oneHotCategorical.to(self.device)

        torchInput = torch.concat((mainEmbeddings, surroundingEmbeddings, embs, normalisedNumerics, oneHotCategorical),
                                  dim=2)

        return torchInput

    def labelTokens(self, seqTokens, goldData):

        tokenLabels = [0 for i in seqTokens]

        labelTranslation = {'invoiceDate': 1, 'invoiceNumber': 2, 'invoiceGrossAmount': 3, 'invoiceTaxAmount': 4,
                            'orderNumber': 5, 'issuerName': 6, 'issuerIBAN': 7, 'issuerAddress': 8, "issuerCity": 9}
        for key, i in goldData.items():
            if i is not None:
                temp = checkSequenceExistence(seqTokens, i)
                if temp[0]:
                    for currentTokenIdx in range(temp[1], temp[1] + len(i)):
                        tokenLabels[currentTokenIdx] = labelTranslation[key]
        return torch.tensor([tokenLabels])

    def forward(self, inputTensor, labels=None):

        inputTensor = inputTensor.to(self.device)
        projectedInput = self.inputProjection(inputTensor)

        outputsAll = self.BERTencoders(projectedInput)
        outputs = outputsAll.last_hidden_state
        emissions = self.classifier(outputs)

        if labels is not None:
            labels = labels.to(self.device)
            loss = -self.crf.forward(emissions, labels, reduction="mean")
            tags = self.crf.decode(emissions)
            return loss, tags
        else:
            return self.crf.decode(emissions)

    def trainModel(self, numEpochs, dataset, trainHistoryPath="", lr=1e-3):

        if trainHistoryPath:
            trainHistory = pd.read_csv(trainHistoryPath)

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)

        epochData = pd.DataFrame(columns=['epoch', 'avgLoss'])
        batchData = pd.DataFrame(columns=['epoch', 'invoiceInstance', "predictions", "goldLabels", 'loss'])

        for epoch in range(numEpochs):
            print(f"Epoch {epoch + 1} / {numEpochs}")

            overallEpochLoss = 0
            shuffledIndices = torch.randperm(len(dataset))

            for i in range(len(dataset)):
                # if i == (len(dataset) // 10):
                #   print(f"current data index: {i}")

                dataInstance = dataset[shuffledIndices[i]]
                pathToInstance = dataInstance["instanceFolderPath"]
                print(i, pathToInstance)
                if trainHistoryPath and f"{pathToInstance}_{epoch}" in trainHistory.values:
                    continue

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
                batchData = batchData.append(
                    {'epoch': epoch + 1, 'invoiceInstance': pathToInstance.split("\\")[-2:], "predictions": tags,
                     "goldLabels": labels.tolist(), 'loss': loss.item()}, ignore_index=True)

                loss.backward()
                optimizer.step()

                overallEpochLoss += loss.item()

                if trainHistoryPath:
                    trainHistory.loc[len(trainHistory)] = f"{pathToInstance}_{epoch}"

            overallEpochLoss = overallEpochLoss / len(dataset)
            print(f"Avg. loss for epoch {epoch + 1}: {overallEpochLoss}")
            epochData = epochData.append({'epoch': epoch + 1, 'avgLoss': overallEpochLoss}, ignore_index=True)

            scheduler.step()

        time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        epochData.to_csv(f"./trainEpochData_{time}.csv")
        batchData.to_csv(f"./trainBatchData_{time}.csv")

        if trainHistoryPath:
            trainHistory.to_csv(trainHistoryPath, index=False)

        print("Training of BERT-CRF complete")

    def testModel(self, dataset):
        testResults = pd.DataFrame(columns=['invoiceInstance', 'prediction', "goldLabels", "instanceLoss"])
        self.eval()

        with torch.no_grad():
            for i in range(len(dataset)):
                dataInstance = dataset[i]
                pathToInstance = dataInstance["instanceFolderPath"]
                preparedInput = self.prepareInput(dataInstance)
                preparedInput = preparedInput.type(dtype=torch.float32)

                instanceSequence = self.getSequence(dataInstance)
                instanceTokens, instanceTokensDict = self.tokenizeSequence(instanceSequence)
                labels = self.labelTokens(instanceTokens,
                                          getGoldData(
                                              os.path.join(dataInstance["instanceFolderPath"], "goldLabels.json"))
                                          )

                loss, tags = self.forward(preparedInput, labels)
                testResults = testResults.append(
                    {'invoiceInstance': pathToInstance.split("\\")[-2:], 'prediction': tags,
                     "goldLabels": labels.tolist(),
                     "instanceLoss": loss.item()}, ignore_index=True)

        time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        testResults.to_csv(f"./testResults_{time}.csv")

        print("Testing of BERT-CRF complete")
        return testResults


if __name__ == "__main__":
    #    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #   print(device)

    data = CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
    invoiceBERT = InvoiceBERT()

    torch.save(invoiceBERT.state_dict(), getConfig("BERT_based", CONFIG_PATH)["pathToStateDict"])

    # invoiceBERT.load_state_dict(torch.load(getConfig("BERT_based", CONFIG_PATH)["pathToStateDict"]))

    # invoiceBERT.trainModel(2, data,
    #                       trainHistoryPath=r"C:\Users\fabia\InvoiceInformationExtraction\BERT_based\trainHistory.csv")
    # invoiceBERT.trainModel(3, data)
    # invoiceBERT.testModel(data)
