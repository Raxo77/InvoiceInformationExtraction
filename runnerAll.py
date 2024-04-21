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
    from transformers import BertTokenizer, BertModel

    # initialise, train and test datasets
    trainData = customDataset.CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
    testData = customDataset.CustomDataset(getConfig("pathToTestDataFolder", CONFIG_PATH))

    # initialise and load Model 1
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    underlyingModel = BertModel.from_pretrained('bert-base-cased')
    numEpochsInvoiceBERT = getConfig("BERT_based", CONFIG_PATH)["numEpochs"]
    learningRateInvoiceBERT = getConfig("BERT_based", CONFIG_PATH)["learningRate"]

    invoiceBERT = BERT_CRF.InvoiceBERT(tokenizer=tokenizer,
                                       model=underlyingModel,
                                       featureSize=getConfig("BERT_based", CONFIG_PATH)["inputFeatureSize"],
                                       numLabels=getConfig("numLabels", CONFIG_PATH)
                                       )
    invoiceBERT.load_state_dict(torch.load(getConfig("BERT_based", CONFIG_PATH)["pathToStateDict"]))
    resListTrainInvoiceBERT = invoiceBERT.trainModel(numEpochs=numEpochsInvoiceBERT,
                                                     dataset=trainData,
                                                     saveResults=True,
                                                     lr=learningRateInvoiceBERT)
    resListTestInvoiceBERT = invoiceBERT.testModel(dataset=testData)
    torch.save(invoiceBERT.state_dict(), getConfig("BERT_based", CONFIG_PATH)["pathToStateDict"])

    # initialise, train and test Model 2
