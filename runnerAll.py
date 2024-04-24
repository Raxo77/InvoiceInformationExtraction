"""
Script to initialise and train all models in isolation.
Subsequently, (trained) models are tested in isolation and in concert as an ensemble learner.
This file will also load all the variables from the config file

Model 1: BERT_based
Model 2: CloudScan_based
Model 3: BBMC_based
Model 4: BiLSTM-CNN-CRF_based
Model 5: GCN_based
Model 6: Intellix_based
"""

if __name__ == '__main__':
    from utils.helperFunctions import getConfig, CONFIG_PATH
    import dataProcessing.customDataset as customDataset
    import BERT_based.BERT_CRF as BERT_CRF
    import torch
    from transformers import BertTokenizerFast, BertModel

    # get device, i.e., GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialise, train and test datasets
    trainData = customDataset.CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
    testData = customDataset.CustomDataset(getConfig("pathToTestDataFolder", CONFIG_PATH))


    def subModel1(train, test):

        # INITIALISE AND LOAD MODEL 1
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
                                           numLabels=numLabelsBERT
                                           )

        invoiceBERT.load_state_dict(torch.load(pathToSavedVersionBERT))

        if train:
            invoiceBERT.trainModel(numEpochs=numEpochsInvoiceBERT,
                                   dataset=trainData,
                                   trainHistoryPath=pathToTrainingHistoryBERT,
                                   lr=learningRateInvoiceBERT)

        if test:
            invoiceBERT.testModel(dataset=testData)

        torch.save(invoiceBERT.state_dict(), pathToSavedVersionBERT)

    # INITIALISE AND LOAD MODEL 2


    subModel1(train=True,test=False)