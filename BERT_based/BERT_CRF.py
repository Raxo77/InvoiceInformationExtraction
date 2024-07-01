import os
import torch
import warnings
import pandas as pd
from torchcrf import CRF
# from TorchCRF import CRF
from datetime import datetime
from transformers import BertTokenizerFast, BertModel
from dataProcessing.customDataset import CustomDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.helperFunctions import loadJSON, getConfig, CONFIG_PATH, createJSON

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
                 numLabels=10,
                 batchSize=8
                 ):
        super(InvoiceBERT, self).__init__()

        self.device = device
        self.featureSize = featureSize
        self.hiddenSize = model.config.hidden_size

        self.inputProjection = torch.nn.Linear(featureSize, self.hiddenSize).to(device)

        self.tokenizer = tokenizer
        self.batchSize = batchSize
        self.embeddingLayer = model.embeddings.to(device)

        self.BERTencoders = model.encoder.to(device)
        self.BERTpooler = model.pooler.to(device)

        self.classifier = torch.nn.Linear(self.hiddenSize, numLabels).to(device)
        self.crf = CRF(numLabels, batch_first=True).to(device)

    def getSequence(self, dataInstance):
        """
        Get the raw text sequence of an invoice document based on the OCR output
        """
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
                seqString += f" {i}"  # if i is no punctuation char, append word with a whitespace

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

    def forward(self, inputTensor, labels=None, attentionMask=None):

        # dim of inputTensor (batchSize, seqLen, originalFeatureLen (2316))
        inputTensor = inputTensor.to(self.device)
        #
        # dim of projected input (batchSize, seqLen, projectedFeatureLen (768))
        projectedInput = self.inputProjection(inputTensor)

        # dim of attention mask here (batchSize, 1, seqLen, seqLen)
        outputsAll = self.BERTencoders(projectedInput, attention_mask=attentionMask)
        outputs = outputsAll.last_hidden_state

        emissions = self.classifier(outputs)

        if labels is not None:
            labels = labels.to(self.device)
            attentionMask = attentionMask[:, 0, 0, :]
            loss = -self.crf.forward(emissions, labels, reduction="mean", mask=attentionMask.bool())
            tags = self.crf.decode(emissions)
            return loss, tags
        else:
            return self.crf.decode(emissions)

    def trainModel(self,
                   numEpochs,
                   dataset,
                   trainHistoryPath="",
                   lr=1e-3):

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
                attentionMaskList = []

                # innermost loop - respectively handles concrete instances in each batch
                for batchNum, idx in enumerate(allInstances):

                    dataInstance = dataset[idx]

                    pathToInstance = dataInstance["instanceFolderPath"]
                    batchData[batchDataIndex]["batchItems"].append(pathToInstance.split("\\")[-1])

                    print(i, batchNum, pathToInstance)
                    itemNum = pathToInstance.split("\\")[-1]

                    if trainHistoryPath and f"{itemNum}_{epoch}" in trainHistory.values:
                        continue

                    if trainHistoryPath:
                        trainHistory.loc[len(trainHistory)] = f"{itemNum}_{epoch}"

                    preparedInput = self.prepareInput(dataInstance)
                    preparedInput = preparedInput.type(dtype=torch.float32)

                    instanceSequence = self.getSequence(dataInstance)
                    instanceTokens, instanceTokensDict = self.tokenizeSequence(instanceSequence)
                    labels = self.labelTokens(instanceTokens,
                                              getGoldData(
                                                  os.path.join(dataInstance["instanceFolderPath"], "goldLabels.json"))
                                              )
                    preparedInputList.append(preparedInput[0])
                    labelsList.append(labels[0])

                    batchData[batchDataIndex]["goldLabels"].append(labels[0].tolist())

                    attentionMaskList.append(torch.ones((1, preparedInput.size(1))))

                # end of innermost loop - i.e. pre-processing for all invoices of resp. batch complete

                if not preparedInputList:
                    print("skipped")
                    continue

                maxBatchLength = max(t.size(0) for t in preparedInputList)

                preparedInputList = torch.stack([torch.nn.functional.pad(t,
                                                                         (0,
                                                                          0,
                                                                          0,
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

                attentionMask = attentionMask.unsqueeze(-1).repeat(1, 1, 1, attentionMask.size(-1)).to(self.device)

                optimizer.zero_grad()
                loss, tags = self.forward(preparedInputList, labelsList, attentionMask=attentionMask)

                for c, j in enumerate(batchData[batchDataIndex]["goldLabels"]):
                    batchData[batchDataIndex]["predictions"].append(tags[c][:len(j)])

                overallEpochLoss += loss.item()
                batchData[batchDataIndex]["batchLoss"] = loss.item()

                loss.backward()
                optimizer.step()

            # end of intermediate loop - respective epoch complete

            createJSON(f"./batchData.json", batchData)

            if trainHistoryPath:
                trainHistory.to_csv(trainHistoryPath, index=False)

            # after each epoch, save current state dict as checkpoint
            time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            checkpointPath = getConfig("BERT_based", CONFIG_PATH)["pathToStateDict"][:-3] + f"{epoch}_"
            checkpointPath += time
            checkpointPath += ".pt"
            torch.save(self.state_dict(), checkpointPath)

            overallEpochLoss = overallEpochLoss / ((len(dataset) // self.batchSize) * self.batchSize)
            epochData = epochData.append({'epoch': epoch + 1, 'avgLoss': overallEpochLoss}, ignore_index=True)
            scheduler.step()

            epochData.to_csv(f"./trainEpochData_06-06.csv", index=False)

        torch.save(self.state_dict(), getConfig("BERT_based", CONFIG_PATH)["pathToStateDict"])
        print("Training of BERT-CRF complete")

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

                preparedInput = self.prepareInput(dataInstance)
                preparedInput = preparedInput.type(dtype=torch.float32)[0]
                preparedInput = preparedInput[None, :, :]

                instanceSequence = self.getSequence(dataInstance)
                instanceTokens, instanceTokensDict = self.tokenizeSequence(instanceSequence)
                labels = self.labelTokens(instanceTokens,
                                          getGoldData(
                                              os.path.join(dataInstance["instanceFolderPath"], "goldLabels.json"))
                                          )
                labels = labels.to(self.device)

                attentionMask = torch.ones((1, preparedInput.size(1)))
                attentionMask = attentionMask.unsqueeze(-1).repeat(1, 1, 1, attentionMask.size(-1)).to(self.device)

                loss, predictions = self.forward(preparedInput, labels, attentionMask=attentionMask)
                overallLoss += loss.item()

                testResults[itemNum]["instanceTokens"] = instanceTokens
                testResults[itemNum]["goldLabels"] = labels[0].tolist()
                testResults[itemNum]["predictions"] = predictions[0]
                testResults[itemNum]["loss"] = loss.item()
                testResults[itemNum]["NumberOfNonZeroLabelsPredicted"] = sum(
                    list(map(lambda x: 1 if x != 0 else 0, predictions[0])))
                testResults[itemNum]["NumberOfNonZeroLabelsGold"] = sum(
                    list(map(lambda x: 1 if x != 0 else 0, labels[0].tolist())))

                testResults[itemNum]["accuracy"] = accuracy_score(labels[0].tolist(),
                                                                  predictions[0])

                overallAcc += testResults[itemNum]["accuracy"]

                testResults[itemNum]["confusionMatrix"] = confusion_matrix(
                    labels[0].tolist(),
                    predictions[0]).tolist()

        time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        createJSON(f"F:\\CodeCopy\\InvoiceInformationExtraction\\BERT_based\\testResults_{time}.json", testResults)
        print("Average Test Loss: {}".format(overallLoss / len(dataset)))
        print("Average Test Accuracy: {}".format(overallAcc / len(dataset)))
        print("-" * 100)
        print("Testing of BERT_CRF complete")

        return testResults


    """
    Model info:
    
    - number of parameters: 110_097_538
    - time to train for 2,000 items with batchSize=8 per epoch:
        ~ 250 minutes 
    """
