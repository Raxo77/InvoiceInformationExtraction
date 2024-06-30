"""
Script to initialise and train all models in isolation.
Subsequently, (trained) models are tested in isolation and in concert as an ensemble learner.
This file will also load all the variables from the config file

Model 1: BERT_based
Model 2: CloudScan_based
Model 3: BBMC_based
Model 4: BiLSTM_CNN_CRF_based
Model 5: GCN_based
Model 6: Intellix_based
"""

import torch
import pandas as pd
import BBMC_based.BBMC as BBMC
import GCN_based.GCN as GCN_based
import BERT_based.BERT_CRF as BERT_CRF
import CloudScan_based.CloudScan as CloudScan_based
import dataProcessing.customDataset as customDataset
from transformers import BertTokenizerFast, BertModel
from utils.helperFunctions import getConfig, CONFIG_PATH
import BiLSTM_CNN_CRF_based.BiLSTM_CNN_CRF as BiLSTM_CNN_CRF
from Intellix_based.templateApproach import TemplateBasedExtraction

torch.manual_seed(123)

# get device, i.e., GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialise, train and test datasets
trainData = customDataset.CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
testData = customDataset.CustomDataset(getConfig("pathToTestDataFolder", CONFIG_PATH))


def subModel1(train, test):
    # INITIALISE AND LOAD BERT-BASED MODEL
    batchSizeBERT = getConfig("BERT_based", CONFIG_PATH)["batchSize"]
    tokenizerBERT = BertTokenizerFast.from_pretrained(getConfig("BERT_based", CONFIG_PATH)["tokenizer"])
    underlyingModelBERT = BertModel.from_pretrained(getConfig("BERT_based", CONFIG_PATH)["model"])
    numEpochsInvoiceBERT = getConfig("BERT_based", CONFIG_PATH)["numEpochs"]
    learningRateInvoiceBERT = getConfig("BERT_based", CONFIG_PATH)["learningRate"]
    inputFeatureSizeBERT = getConfig("BERT_based", CONFIG_PATH)["inputFeatureSize"]
    pathToSavedVersionBERT = getConfig("BERT_based", CONFIG_PATH)["pathToStateDict"]
    pathToTrainingHistoryBERT = getConfig("BERT_based", CONFIG_PATH)["pathToTrainingHistory"]
    numLabelsBERT = getConfig("BERT_based", CONFIG_PATH)["numLabels"]

    invoiceBERT = BERT_CRF.InvoiceBERT(device=device,
                                       tokenizer=tokenizerBERT,
                                       model=underlyingModelBERT,
                                       featureSize=inputFeatureSizeBERT,
                                       numLabels=numLabelsBERT,
                                       batchSize=batchSizeBERT
                                       )

    print("passed01")
    invoiceBERT.load_state_dict(torch.load(pathToSavedVersionBERT))
    print("passed02")

    if train:
        invoiceBERT.trainModel(numEpochs=numEpochsInvoiceBERT,
                               dataset=trainData,
                               trainHistoryPath=pathToTrainingHistoryBERT,
                               lr=learningRateInvoiceBERT)

    if test:
        testResults = invoiceBERT.testModel(dataset=testData)

    return testResults if test else None


def subModel2(test, train):
    # INITIALISE AND LOAD CloudScan-BASED MODEL
    citiesGazetteer = pd.read_csv(getConfig("pathToCityGazetteer", CONFIG_PATH), sep="\t")
    print("Gazetteers loaded")
    countryGazetteer = pd.read_csv(getConfig("pathToCountryGazetteer", CONFIG_PATH))
    print("Gazetteers loaded")
    ZIPCodesGazetteer = None  # pd.read_csv(getConfig("pathToZIPGazetteer", CONFIG_PATH), header=None, sep="\t")
    print("Gazetteers loaded")
    batchSizeCloudScan = getConfig("CloudScan_based", CONFIG_PATH)["batchSize"]
    numEpochsCloudScan = getConfig("CloudScan_based", CONFIG_PATH)["numEpochs"]
    hashSizeCloudScan = getConfig("CloudScan_based", CONFIG_PATH)["hashSize"]
    embeddingSizeCloudScan = getConfig("CloudScan_based", CONFIG_PATH)["embeddingSize"]
    learningRateCloudScan = getConfig("CloudScan_based", CONFIG_PATH)["learningRate"]
    inputFeatureSizeCloudScan = getConfig("CloudScan_based", CONFIG_PATH)["inputFeatureSize"]
    pathToSavedVersionCloudScan = getConfig("CloudScan_based", CONFIG_PATH)["pathToStateDict"]
    pathToTrainingHistoryCloudScan = getConfig("CloudScan_based", CONFIG_PATH)["pathToTrainingHistory"]
    numLabelsCloudScan = getConfig("CloudScan_based", CONFIG_PATH)["numLabels"]

    cloudScan = CloudScan_based.CloudScanLSTM(hashSize=hashSizeCloudScan,
                                              embeddingSize=embeddingSizeCloudScan,
                                              inputSize=inputFeatureSizeCloudScan,
                                              numLabels=numLabelsCloudScan,
                                              citiesGazetteer=citiesGazetteer,
                                              batchSize=batchSizeCloudScan,
                                              countriesGazetteer=countryGazetteer,
                                              ZIPgazetteer=ZIPCodesGazetteer,
                                              device=device
                                              )

    print("passed01")
    cloudScan.load_state_dict(torch.load(pathToSavedVersionCloudScan))
    print("passed02")

    if train:
        cloudScan.trainModel(numEpochs=numEpochsCloudScan,
                             dataset=trainData,
                             trainHistoryPath=pathToTrainingHistoryCloudScan,
                             lr=learningRateCloudScan)

    if test:
        testResults = cloudScan.testModel(dataset=testData)

    return testResults if test else None


def subModel3(train, test):
    pass

    # INITIALISE AND LOAD BBMC-BASED MODEl

    lrBBMC = getConfig("BBMC_based", CONFIG_PATH)["learningRate"]
    tokenizerBBMC = BertTokenizerFast.from_pretrained(getConfig("BBMC_based", CONFIG_PATH)["tokenizer"])
    modelBBMC = BertModel.from_pretrained(getConfig("BBMC_based", CONFIG_PATH)["model"])
    hiddenDimBBMC = getConfig("BBMC_based", CONFIG_PATH)["hiddenDim"]
    LSTMlayersBBMC = getConfig("BBMC_based", CONFIG_PATH)["LSTMlayers"]
    dropoutRateBBMC = getConfig("BBMC_based", CONFIG_PATH)["dropoutRate"]
    numLabelsBBMC = getConfig("BBMC_based", CONFIG_PATH)["numLabels"]
    batchSizeBBMC = getConfig("BBMC_based", CONFIG_PATH)["batchSize"]
    numEpochsBBMC = getConfig("BBMC_based", CONFIG_PATH)["numEpochs"]
    trainHistoryPathBiLSTM = getConfig("BBMC_based", CONFIG_PATH)["pathToTrainingHistory"]
    pathToSavedVersionBiLSTM = getConfig("BBMC_based", CONFIG_PATH)["pathToStateDict"]

    invoice_BBMC_based = BBMC.InvoiceBBMC(device=device,
                                          tokenizer=tokenizerBBMC,
                                          model=modelBBMC,
                                          hiddenDim=hiddenDimBBMC,
                                          LSTMlayers=LSTMlayersBBMC,
                                          dropoutRate=dropoutRateBBMC,
                                          numLabels=numLabelsBBMC,
                                          batchSize=batchSizeBBMC
                                          )
    print("passed01")
    invoice_BBMC_based.load_state_dict(torch.load(pathToSavedVersionBiLSTM))
    print("passed02")

    if train:
        invoice_BBMC_based.trainModel(numEpochs=numEpochsBBMC,
                                      dataset=trainData,
                                      trainHistoryPath=trainHistoryPathBiLSTM,
                                      lr=lrBBMC)

    if test:
        testResults = invoice_BBMC_based.testModel(dataset=testData)

    return testResults if test else None


def subModel4(train, test):
    pass

    # INITIALISE AND LOAD BILSTM-CNN-CRF-BASED MODEl

    # Training with fastText as embeddings
    lrBiLSTM = getConfig("BiLSTM_CNN_CRF_based", CONFIG_PATH)["learningRate"]
    wordEmbeddingStoiPathBiLSTM = getConfig("pathToFastTextStoi", CONFIG_PATH)
    batchSizeBiLSTM = getConfig("BiLSTM_CNN_CRF_based", CONFIG_PATH)["batchSize"]
    numEpochsBiLSTM = getConfig("BiLSTM_CNN_CRF_based", CONFIG_PATH)["numEpochs"]
    kernelSizeBiLSTM = getConfig("BiLSTM_CNN_CRF_based", CONFIG_PATH)["kernelSize"]
    wordEmbeddingVectorsPathBiLSTM = getConfig("pathToFastTextVectors", CONFIG_PATH)
    pathToSavedVersionBiLSTM = getConfig("BiLSTM_CNN_CRF_based", CONFIG_PATH)["pathToStateDict"]
    charEmbeddingSizeBiLSTM = getConfig("BiLSTM_CNN_CRF_based", CONFIG_PATH)["charEmbeddingSize"]
    maxWordsPerInvoiceBiLSTM = getConfig("BiLSTM_CNN_CRF_based", CONFIG_PATH)["maxWordsPerInvoice"]
    trainHistoryPathBiLSTM = getConfig("BiLSTM_CNN_CRF_based", CONFIG_PATH)["pathToTrainingHistory"]
    trainableEmbeddingsBiLSTM = getConfig("BiLSTM_CNN_CRF_based", CONFIG_PATH)["trainableEmbeddings"]

    Invoice_BiLSTM_CNN_CRF_based = BiLSTM_CNN_CRF.Invoice_BiLSTM_CNN_CRF(device=device,
                                                                         wordEmbeddingVectorsPath=wordEmbeddingVectorsPathBiLSTM,
                                                                         wordEmbeddingStoiPath=wordEmbeddingStoiPathBiLSTM,
                                                                         maxWordsPerInvoice=maxWordsPerInvoiceBiLSTM,
                                                                         charEmbeddingSize=charEmbeddingSizeBiLSTM,
                                                                         kernelSize=kernelSizeBiLSTM,
                                                                         trainableEmbeddings=trainableEmbeddingsBiLSTM,
                                                                         batchSize=batchSizeBiLSTM
                                                                         )

    print("passed01")
    Invoice_BiLSTM_CNN_CRF_based.load_state_dict(torch.load(pathToSavedVersionBiLSTM))
    print("passed02")

    if train:
        Invoice_BiLSTM_CNN_CRF_based.trainModel(dataset=trainData,
                                                numEpochs=numEpochsBiLSTM,
                                                trainHistoryPath=trainHistoryPathBiLSTM,
                                                lr=lrBiLSTM
                                                )

    if test:
        testResults = Invoice_BiLSTM_CNN_CRF_based.testModel(dataset=testData)

    return testResults if test else None


def subModel5(train, test):
    # INITIALISE AND LOAD GCN-BASED MODEl

    numLabelsGCN = getConfig("GCN_based", CONFIG_PATH)["numLabels"]
    tokenizerGCN = BertTokenizerFast.from_pretrained(getConfig("GCN_based", CONFIG_PATH)["tokenizer"])
    underlyingModelGCN = BertModel.from_pretrained(getConfig("GCN_based", CONFIG_PATH)["model"])
    numFiltersGCN = getConfig("GCN_based", CONFIG_PATH)["initialNumFilters"]
    filterSizeGCN = getConfig("GCN_based", CONFIG_PATH)["filterSize"]
    featureSizeGCN = getConfig("GCN_based", CONFIG_PATH)["featureSize"]
    batchSizeGCN = getConfig("GCN_based", CONFIG_PATH)["batchSize"]
    pathToSavedVersionGCN = getConfig("GCN_based", CONFIG_PATH)["pathToStateDict"]
    trainHistoryPathGCN = getConfig("GCN_based", CONFIG_PATH)["pathToTrainingHistory"]
    numEpochsGCN = getConfig("GCN_based", CONFIG_PATH)["numEpochs"]
    lrGCN = getConfig("GCN_based", CONFIG_PATH)["learningRate"]
    saveInvoiceGraphGCN = getConfig("GCN_based", CONFIG_PATH)["saveGraph"]

    invoiceGCN = GCN_based.InvoiceGCN(numClasses=numLabelsGCN,
                                      model=underlyingModelGCN,
                                      tokenizer=tokenizerGCN,
                                      device=device,
                                      initFilterNumber=numFiltersGCN,
                                      filterSize=filterSizeGCN,
                                      featureSize=featureSizeGCN,
                                      batchSize=batchSizeGCN)
    print("passed01")
    invoiceGCN.load_state_dict(torch.load(pathToSavedVersionGCN))
    print("passed02")

    if train:
        invoiceGCN.trainModel(numEpochs=numEpochsGCN,
                              dataset=trainData,
                              trainHistoryPath=trainHistoryPathGCN,
                              lr=lrGCN,
                              saveInvoiceGraph=saveInvoiceGraphGCN)

    if test:
        testResults = invoiceGCN.testModel(dataset=testData)

    return testResults if test else None


def subModel6(employ, referenceDataset, employmentDataset):

    # INITIALISE AND LOAD INTELLIX-BASED MODEl

    pathToPeersTemplate = getConfig("Intellix_based", CONFIG_PATH)["pathToPeers"]
    pathToScriptsTemplate = getConfig("Intellix_based", CONFIG_PATH)["pathToScripts"]

    templateBasedExtractor = TemplateBasedExtraction(pathToPeers=pathToPeersTemplate,
                                                     pathToScripts=pathToScriptsTemplate,
                                                     referenceDataset=referenceDataset)
    if employ:
        testResults = templateBasedExtractor.testTemplateApproach(employmentDataset)

        return testResults


if __name__ == '__main__':

    # subModel1(train=True, test=True)
    # subModel2(train=True, test=True)
    # subModel3(train=True, test=True)
    # subModel4(train=True, test=True)
    # subModel5(train=True, test=True)
    #subModel6(employ=True, referenceDataset=trainData, employmentDataset=testData)
    pass
