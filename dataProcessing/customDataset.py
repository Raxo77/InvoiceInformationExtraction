from torch.utils.data import Dataset
import os
from bs4 import BeautifulSoup
from filterRawDataset import listDirectory, PATH_TO_DATA_FOLDER


class CustomDataset(Dataset):
    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.instances = listDirectory(rootDir)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, instanceIdx):
        """

        :param instanceIdx:
        :return: dict with hOCR result and information to derive further features if desired
        """

        instanceFolderPath = self.instances[instanceIdx].path
        instanceFolderContent = listDirectory(instanceFolderPath, folderOnly=False)

        with open(os.path.join(instanceFolderPath.path, "hOCR_output.xml"), 'r', encoding='utf-8') as f:
            hOCR = f.read()

        data = {"pathInvoicePDF": os.path.join(instanceFolderPath, "flat_document.pdf"),
                "pathInvoicePNG": os.path.join(instanceFolderPath, "flat_document.png"),
                "hOCR": BeautifulSoup(hOCR, features="lxml")
                }

        return data
