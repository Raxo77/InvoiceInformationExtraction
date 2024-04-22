import os
from utils.helperFunctions import getConfig, separate, loadJSON, createJSON, CONFIG_PATH

""""
The dataset used in the scope of this thesis is the Inv3D Dataset (https://felixhertlein.github.io/inv3d/)
which contains 25,000 invoice images over 100 templates in both .pdf and .png formats alongside invoice metadata and ground
truth entity labels. Invoices were created by filling 100 HTML templates with fully randomised content.
The dataset was created in the context of a research paper on document unwarping:
@article{Hertlein2023,
  title        = {Inv3D: a high-resolution 3D invoice dataset for template-guided single-image document unwarping},
  author       = {Hertlein, Felix and Naumann, Alexander and Philipp, Patrick},
  year         = 2023,
  month        = {Apr},
  day          = 29,
  journal      = {International Journal on Document Analysis and Recognition (IJDAR)},
  doi          = {10.1007/s10032-023-00434-x},
  ISSN         = {1433-2825},
  url          = {https://doi.org/10.1007/s10032-023-00434-x}
}

Saved locally, the entire dataset - downloadable in pre-defined train, validation and test splits - has about 860GB.
However, as not all the information provided per invoice are needed throughout this thesis, a first step comprises
the filtering for relevant information to allow for more streamlined down-the-line processing.  
"""

CONFIG = CONFIG_PATH
PATH_TO_DATA_FOLDER = getConfig("pathToDataFolder", CONFIG)
FILES_TO_DELETE = getConfig("filesToDelete", CONFIG)


def listDirectory(path: str = PATH_TO_DATA_FOLDER, folderOnly: bool = True) -> list:
    # Returns list of all folders/files contained immediately (i.e., no subfolders/-files) within a directory
    dirList = list(os.scandir(path))
    if folderOnly:
        return [i for i in dirList if i.is_dir()]
    else:
        return dirList


def filterFiles(dirList: list, blacklist: list[str] = FILES_TO_DELETE, feedback: bool = False,
                saveDeletions: str = "") -> None:
    """
    Deletes all files/folders contained within ´blacklist´ from each directory in dirList

    :param dirList: list of <class 'nt.DirEntry'> containing folders which are to be filtered
    :param blacklist: list of strings containing the names of the files/folders which are to be deleted
    :param feedback: boolean governing whether printed feedback is shown (printed if True)
    :param saveDeletions: string of the path where to store information on deleted element per folder
    :return: None (indirectly the JSON of the deletion information if saveDeletions is given and valid)
    """

    deletionInfo = {}
    for count, directory in enumerate(dirList):
        deletionCount = 0
        deletionInfo[f"{directory.name}_{count}"] = {"directoryPath": directory.path,
                                                     "deletedElements": [],
                                                     "sumDeletedElements": 0
                                                     }
        for file in listDirectory(directory.path, folderOnly=False):
            if file.name in blacklist:
                os.remove(file.path)

                deletionInfo[f"{directory.name}_{count}"]["deletedElements"].append((file.name, file.path))
                deletionCount += 1
                if feedback:
                    print(f"removed {file.name} from {file.path}")
        deletionInfo[f"{directory.name}_{count}"]["sumDeletedElements"] = deletionCount
        if feedback:
            separate()
    if saveDeletions:
        if os.path.exists(saveDeletions):
            # if the provided deletion path exists, alert and potentially gets new path - notably, checks only once
            print(f"WARNING: deletionInformation.json under the path ´{saveDeletions}´ already exists.")
            temp = input("Enter ´y´ to continue anyway or give new saveDeletions path:")
            if temp != "y":
                saveDeletions = temp
        createJSON(saveDeletions, deletionInfo)
        if feedback:
            print(f"Information of deleted files saved under {saveDeletions}")


def getGoldLabels(pathToJSON: str, targets: dict, dirPath: str = "") -> dict:
    """
    Extracts the entity labels relevant in the scope of this thesis and subsequent algorithms from an invoice's
    ´ground_truth_tags.json´. Notably, the underlying JSON must be structured the same way as ´ground_truth_tags.json´.
    Returns and optionally stores them as a dict of dicts containing each tags value, x,y bbox
    coordinates, height and width.

    :param pathToJSON: path to the ground_truth_tags.json file
    :param targets: a dict of the tag name and a list containing the respective parts of the label. That is to say, tags
    such as ´address´ may be comprised of multiple individual tokens such as ´city´ and ´country´.
    :param dirPath: the respective root path where to store the extracted labels, if desired. If not provided, labels
    are only returned. If dirPath == <same>, the root path of pathToJSON is taken.
    :return: a dict of the respective gold labels and their metadata
    """

    data = loadJSON(pathToJSON)

    data = {
        entry["tag"]: {k: v for k, v in entry.items() if k != "tag"}
        for entry in data
    }
    goldLabels = {i: [] for i in targets.keys()}
    for k, v in targets.items():
        for i in v:
            try:
                goldLabels[k].append(data[i])
            except KeyError:
                goldLabels[k].append(None)
    temp = goldLabels
    goldLabels = {}
    for k, v in temp.items():

        if len(v) > 0:
            goldLabels[k] = v[0]
        else:
            goldLabels[k] = None

    if dirPath == "<same>":
        dirPath = pathToJSON[:-len("ground_truth_tags.json")]

    if dirPath:
        createJSON(f"{dirPath}\\goldLabels.json", goldLabels)

    return goldLabels


if __name__ == '__main__':
    pass