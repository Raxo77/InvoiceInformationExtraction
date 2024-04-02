from PIL import Image
from utils.CONFIG_PATH import CONFIG_PATH
from utils.helperFunctions import getConfig
import pandas as pd
import numpy as np
import re
from dateutil.parser import parse, ParserError
from dataProcessing.customDataset import CustomDataset


def featureCalculation(nGramList: list, dataInstance):
    """
    :param nGramList: a list of sets of ngrams for the corresponding invoice
    :param dataInstance: the concrete dataInstance for which to deriveFeatures; cf. CustomDataset
    :return:
    """
    # (focalWord | stringFeatures | numericFeatures | booleanFeatures | wholeNGram)
    # for each element in the final list
    nGramFeatures = []

    featuresDF = pd.read_csv(dataInstance["BERT-basedFeaturesPath"])
    colNames = list(featuresDF.columns)
    colNames[0] = "wordKey"
    featuresDF.columns = colNames

    # get tuple of image width, height
    imgWidth, imgHeight = Image.open(dataInstance["pathInvoicePNG"]).size

    # calculate the features as described in CloudScanPaper
    # nGramList is a list of sets containing ngram, focal word pairs
    for wordSet in nGramList:
        nGram, word = wordSet
        focalFeatures = (word, [], [], [], nGram)

        rawText = word.split("_")[0]
        focalFeatures[1].append(rawText)

        rawTextLastWord = nGram[-1].split("_")[0]
        focalFeatures[1].append(rawTextLastWord)

        rawTextTwoWordsLeft = None
        nextWordLeft = featuresDF.loc[featuresDF["wordKey"] == word, "left"].item()
        contextCount = 1
        while nextWordLeft is not np.nan and contextCount < 2:
            nextWordLeft = featuresDF.loc[featuresDF["wordKey"] == nextWordLeft, "left"].item()
            contextCount += 1
        if contextCount == 2 and nextWordLeft is not np.nan:
            rawTextTwoWordsLeft = nextWordLeft
        focalFeatures[1].append(rawTextTwoWordsLeft)

        textPatterns = featuresDF.loc[featuresDF["wordKey"] == word, "standardisedText"].item()
        focalFeatures[1].append(textPatterns)

        # when splitting *word* by "_" - [0] is word; [1] is x coords, [2] is y coords
        bottomMargin = (imgHeight - int(word.split("_")[2]) + int(
            featuresDF.loc[featuresDF["wordKey"] == word, "wordHeight"].item())) / imgHeight
        focalFeatures[2].append(bottomMargin)
        topMargin = int(word.split("_")[2]) / imgHeight
        focalFeatures[2].append(topMargin)
        rightMargin = (imgWidth - int(word.split("_")[1]) - int(
            featuresDF.loc[featuresDF["wordKey"] == word, "wordWidth"].item())) / imgWidth
        focalFeatures[2].append(rightMargin)
        leftMargin = int(word.split("_")[1]) / imgWidth
        focalFeatures[2].append(leftMargin)

        # if a surrounding word exists, take the corresponding values to calculate; else distance is -100:
        bottomMarginRelative = -100
        if featuresDF.loc[featuresDF["wordKey"] == word, "below"].item() is not np.nan:
            bottomMarginRelative = (int(
                featuresDF.loc[featuresDF["wordKey"] == word, "below"].item().split("_")[2]) - int(
                word.split("_")[2]) + int(
                featuresDF.loc[featuresDF["wordKey"] == word, "wordHeight"].item())) / imgHeight
        focalFeatures[2].append(bottomMarginRelative)

        topMarginRelative = -100
        if featuresDF.loc[featuresDF["wordKey"] == word, "above"].item() is not np.nan:
            topMarginRelative = (int(word.split("_")[2]) -
                                 int(featuresDF.loc[featuresDF["wordKey"] == word, "above"].item().split("_")[
                                         2])) / imgHeight
        focalFeatures[2].append(topMarginRelative)

        rightMarginRelative = -100
        if featuresDF.loc[featuresDF["wordKey"] == word, "right"].item() is not np.nan:
            rightMarginRelative = (int(
                featuresDF.loc[featuresDF["wordKey"] == word, "right"].item().split("_")[1]) - int(
                word.split("_")[1]) - int(
                featuresDF.loc[featuresDF["wordKey"] == word, "wordWidth"].item())) / imgWidth
        focalFeatures[2].append(rightMarginRelative)

        leftMarginRelative = -100
        if featuresDF.loc[featuresDF["wordKey"] == word, "left"].item() is not np.nan:
            leftMarginRelative = (int(word.split("_")[1]) -
                                  int(featuresDF.loc[featuresDF["wordKey"] == word, "left"].item().split("_")[
                                          1])) / imgWidth
        focalFeatures[2].append(leftMarginRelative)


        horizontalPosition = -100
        if rightMarginRelative != -100 and leftMarginRelative != -100:
            horizontalPosition = (leftMarginRelative * imgWidth) / (
                    (leftMarginRelative + rightMarginRelative) * imgWidth)

        verticalPosition = -100
        if topMarginRelative != -100 and bottomMarginRelative != -100:
            verticalPosition = (topMarginRelative * imgHeight) / (
                    (topMarginRelative + bottomMarginRelative) * imgHeight)

        focalFeatures[2].append(horizontalPosition)
        focalFeatures[2].append(verticalPosition)

        hasDigits = "d" in textPatterns
        focalFeatures[3].append(hasDigits)


        isKnownCountry = 0
        focalFeatures[3].append(isKnownCountry)
        isKNownZip = 0
        focalFeatures[3].append(isKNownZip)

        # PLACEHOLDER INSERTION - the actual calculation of this feature is performed further below
        leftAlignment = 0
        focalFeatures[2].append(leftAlignment)

        nGramLength = len(" ".join(nGram))
        focalFeatures[2].append(nGramLength)

        pageHeight = imgHeight
        focalFeatures[2].append(pageHeight)

        pageWidth = imgWidth
        focalFeatures[2].append(pageWidth)

        # start with the word itself as linesize
        lineSize = 1
        # increment lineSize as long as focalWord/focalNGram has left and right neighbour
        wordsLeft = 0
        nextWord = featuresDF.loc[featuresDF["wordKey"] == word, "left"].item()
        while nextWord is not np.nan:
            wordsLeft += 1
            nextWord = featuresDF.loc[featuresDF["wordKey"] == nextWord, "left"].item()

        wordsRight = len(nGram) - 1
        nextWord = featuresDF.loc[featuresDF["wordKey"] == nGram[-1], "right"].item()
        while nextWord is not np.nan:
            wordsRight += 1
            nextWord = featuresDF.loc[featuresDF["wordKey"] == nextWord, "right"].item()
        lineSize = lineSize + wordsLeft + wordsRight
        focalFeatures[2].append(lineSize)

        positionOnLine = wordsLeft / lineSize
        focalFeatures[2].append(positionOnLine)

        # Employ top-down approach: get whole area of line and subtract are of all words
        # --> for height of nGram get max height of all words enclosed within
        nGramHeight = [featuresDF.loc[featuresDF["wordKey"] == word, "wordHeight"].item() for word in nGram]
        nGramWidth = [featuresDF.loc[featuresDF["wordKey"] == word, "wordWidth"].item() for word in nGram]
        lineArea = max(nGramHeight) * imgWidth
        nGramArea = sum([h * w for h, w in zip(nGramHeight, nGramWidth)])
        lineWhiteSpace = 1 - (nGramArea / lineArea)
        focalFeatures[2].append(lineWhiteSpace)

        amountPattern = re.compile(r"^\$?\(?\-?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?\)?$")
        parsesAsAmount = bool(amountPattern.match(word.split("_")[0]))
        focalFeatures[3].append(parsesAsAmount)

        parsesAsDate = 0
        try:
            parse(word.split("_")[0])
            parsesAsDate = 1
        except ParserError:
            pass
        focalFeatures[3].append(parsesAsDate)

        parsesAsNumber = 0
        try:
            float(word.split("_")[0])
            parsesAsNumber = 1
        except ValueError:
            pass
        focalFeatures[3].append(parsesAsNumber)

        # instead of math.features get features from BERT_based
        isSum = featuresDF.loc[featuresDF["wordKey"] == word, "isSum"].item()
        focalFeatures[3].append(isSum)
        isMathElement = featuresDF.loc[featuresDF["wordKey"] == word, "isMathElement"].item()
        focalFeatures[3].append(isMathElement)
        isProduct = featuresDF.loc[featuresDF["wordKey"] == word, "isProduct"].item()
        focalFeatures[3].append(isProduct)

        nGramFeatures.append(focalFeatures)

    return nGramFeatures


if __name__ == "__main__":
    data = CustomDataset(getConfig("pathToDataFolder", CONFIG_PATH))
    ngraml = [(['INVOICE_117_157'], 'INVOICE_117_157'),
              (['Bill_117_234', 'From_188_235'], 'Bill_117_234'),
              (['From_188_235'], 'From_188_235'),
              (['Lisa_117_286', 'Montoya,_184_286', 'Yates,_339_286', 'Taylor_432_286'],
               'Lisa_117_286'),
              (['Montoya,_184_286', 'Yates,_339_286', 'Taylor_432_286', 'and_531_286'],
               'Montoya,_184_286'),
              (['Yates,_339_286', 'Taylor_432_286', 'and_531_286', 'Wood_596_286'],
               'Yates,_339_286'),
              (['Taylor_432_286', 'and_531_286', 'Wood_596_286'], 'Taylor_432_286'),
              (['and_531_286', 'Wood_596_286'], 'and_531_286'),
              (['Wood_596_286'], 'Wood_596_286'),
              (['5289_117_327', 'Morgan_201_327', 'Walks_323_327'], '5289_117_327'),
              (['Morgan_201_327', 'Walks_323_327'], 'Morgan_201_327'),
              (['Walks_323_327'], 'Walks_323_327'),
              ([',_421_344'], ',_421_344'),
              (['East_438_327', 'Melissa,_509_327', 'KY_634_327', '34561_682_327'],
               'East_438_327'),
              (['Melissa,_509_327', 'KY_634_327', '34561_682_327'], 'Melissa,_509_327'),
              (['KY_634_327', '34561_682_327'], 'KY_634_327'),
              (['34561_682_327'], '34561_682_327'),
              (['325-285-0965_117_369'], '325-285-0965_117_369'),
              (['Bill_117_492', 'To_188_493'], 'Bill_117_492'),
              (['To_188_493'], 'To_188_493'),
              (['John_114_544', 'Lambert_194_544'], 'John_114_544'),
              (['Lambert_194_544'], 'Lambert_194_544'),
              ([',_325_561'], ',_325_561'),
              (['Johnson_339_544', 'LLC_471_544'], 'Johnson_339_544'),
              (['LLC_471_544'], 'LLC_471_544'),
              (['1216_118_586', 'Suarez_200_586', 'Tunnel_309_586', 'Apt._415_586'],
               '1216_118_586'),
              (['Suarez_200_586', 'Tunnel_309_586', 'Apt._415_586', '333_488_586'],
               'Suarez_200_586'),
              (['Tunnel_309_586', 'Apt._415_586', '333_488_586'], 'Tunnel_309_586'),
              (['Apt._415_586', '333_488_586'], 'Apt._415_586'),
              (['333_488_586'], '333_488_586'),
              ([',_552_603'], ',_552_603'),
              (['South_117_627', 'Victoriachester,_216_627', 'GA_455_627', '93458_508_627'],
               'South_117_627'),
              (['Victoriachester,_216_627', 'GA_455_627', '93458_508_627'],
               'Victoriachester,_216_627'),
              (['GA_455_627', '93458_508_627'], 'GA_455_627'),
              (['93458_508_627'], '93458_508_627'),
              (['+962778250152_117_669'], '+962778250152_117_669'),
              (['Description_233_859'], 'Description_233_859'),
              (['1421_118_938', 'Pink_235_938', 'blue_318_938', 'felt_401_938'],
               '1421_118_938'),
              (['Pink_235_938', 'blue_318_938', 'felt_401_938', 'craft_472_938'],
               'Pink_235_938'),
              (['blue_318_938', 'felt_401_938', 'craft_472_938', 'trinket_559_938'],
               'blue_318_938'),
              (['felt_401_938', 'craft_472_938', 'trinket_559_938', 'box_682_938'],
               'felt_401_938'),
              (['craft_472_938', 'trinket_559_938', 'box_682_938'], 'craft_472_938'),
              (['trinket_559_938', 'box_682_938'], 'trinket_559_938'),
              (['box_682_938'], 'box_682_938'),
              (['Terms_225_1587', 'and_336_1586', 'conditions_407_1586'], 'Terms_225_1587'),
              (['and_336_1586', 'conditions_407_1586'], 'and_336_1586'),
              (['conditions_407_1586'], 'conditions_407_1586'),
              (['Payment_117_1688', '90_254_1688', 'days_300_1688', 'after_375_1688'],
               'Payment_117_1688'),
              (['90_254_1688', 'days_300_1688', 'after_375_1688', 'invoice_454_1688'],
               '90_254_1688'),
              (['days_300_1688', 'after_375_1688', 'invoice_454_1688', 'date_565_1688'],
               'days_300_1688'),
              (['after_375_1688', 'invoice_454_1688', 'date_565_1688'], 'after_375_1688'),
              (['invoice_454_1688', 'date_565_1688'], 'invoice_454_1688'),
              (['date_565_1688'], 'date_565_1688'),
              (['Invoice:_1311_538', '#898146_1430_538'], 'Invoice:_1311_538'),
              (['#898146_1430_538'], '#898146_1430_538'),
              (['Invoice_1199_580', 'date:_1309_580'], 'Invoice_1199_580'),
              (['date:_1309_580'], 'date:_1309_580'),
              (['06/24/2021_1390_579'], '06/24/2021_1390_579'),
              (['Due_1240_621', 'date:_1309_621'], 'Due_1240_621'),
              (['date:_1309_621'], 'date:_1309_621'),
              (['09/22/2021_1390_620'], '09/22/2021_1390_620'),
              (['Qty_978_855'], 'Qty_978_855'),
              (['Price_1196_859', 'Total_1499_859'], 'Price_1196_859'),
              (['Total_1499_859'], 'Total_1499_859'),
              (['2_999_938', '2.46_1205_938', '4.92_1515_938'], '2_999_938'),
              (['2.46_1205_938', '4.92_1515_938'], '2.46_1205_938'),
              (['4.92_1515_938'], '4.92_1515_938'),
              (['Subtotal_1194_1172'], 'Subtotal_1194_1172'),
              (['4.92_1515_1176'], '4.92_1515_1176'),
              (['Sales_1086_1255', 'Tax_1187_1256'], 'Sales_1086_1255'),
              (['Tax_1187_1256'], 'Tax_1187_1256'),
              (['5.4%_1257_1255', '0.27_1516_1255'], '5.4%_1257_1255'),
              (['0.27_1516_1255'], '0.27_1516_1255'),
              (['Shipping_992_1334', '&_1151_1334', 'Handling_1189_1334', '5.65_1516_1334'],
               'Shipping_992_1334'),
              (['&_1151_1334', 'Handling_1189_1334', '5.65_1516_1334'], '&_1151_1334'),
              (['Handling_1189_1334', '5.65_1516_1334'], 'Handling_1189_1334'),
              (['5.65_1516_1334'], '5.65_1516_1334'),
              (['Total_1176_1413', 'Due_1274_1414'], 'Total_1176_1413'),
              (['Due_1274_1414'], 'Due_1274_1414'),
              (['$_1462_1409'], '$_1462_1409'),
              (['10.84_1495_1413'], '10.84_1495_1413'),
              (['Please_1010_1586', 'make_1128_1586', 'a_1226_1594'], 'Please_1010_1586'),
              (['make_1128_1586', 'a_1226_1594'], 'make_1128_1586'),
              (['a_1226_1594'], 'a_1226_1594'),
              (['payment_1254_1589', 'to_1409_1589'], 'payment_1254_1589'),
              (['to_1409_1589'], 'to_1409_1589'),
              (['Beneficiary_1106_1688',
                'Name:_1277_1688',
                'Justin_1374_1688',
                'Watson_1472_1688'],
               'Beneficiary_1106_1688'),
              (['Name:_1277_1688', 'Justin_1374_1688', 'Watson_1472_1688'],
               'Name:_1277_1688'),
              (['Justin_1374_1688', 'Watson_1472_1688'], 'Justin_1374_1688'),
              (['Watson_1472_1688'], 'Watson_1472_1688'),
              (['Beneficiary_1002_1730',
                'Account_1171_1730',
                'Number:_1302_1730',
                '63308297_1436_1730'],
               'Beneficiary_1002_1730'),
              (['Account_1171_1730', 'Number:_1302_1730', '63308297_1436_1730'],
               'Account_1171_1730'),
              (['Number:_1302_1730', '63308297_1436_1730'], 'Number:_1302_1730'),
              (['63308297_1436_1730'], '63308297_1436_1730'),
              (['Bank_1223_1771', 'Name_1304_1771', 'and_1395_1771', 'Address:_1456_1771'],
               'Bank_1223_1771'),
              (['Name_1304_1771', 'and_1395_1771', 'Address:_1456_1771'], 'Name_1304_1771'),
              (['and_1395_1771', 'Address:_1456_1771'], 'and_1395_1771'),
              (['Address:_1456_1771'], 'Address:_1456_1771'),
              (['Ward,_1206_1813',
                'Esparza_1302_1813',
                'and_1422_1813',
                'Krause_1485_1813'],
               'Ward,_1206_1813'),
              (['Esparza_1302_1813', 'and_1422_1813', 'Krause_1485_1813'],
               'Esparza_1302_1813'),
              (['and_1422_1813', 'Krause_1485_1813'], 'and_1422_1813'),
              (['Krause_1485_1813'], 'Krause_1485_1813'),
              (['174_1125_1855',
                'Donna_1189_1855',
                'Trafficway_1296_1855',
                'Apt._1456_1855'],
               '174_1125_1855'),
              (['Donna_1189_1855',
                'Trafficway_1296_1855',
                'Apt._1456_1855',
                '944_1529_1855'],
               'Donna_1189_1855'),
              (['Trafficway_1296_1855', 'Apt._1456_1855', '944_1529_1855'],
               'Trafficway_1296_1855'),
              (['Apt._1456_1855', '944_1529_1855'], 'Apt._1456_1855'),
              (['944_1529_1855'], '944_1529_1855'),
              (['Shawnchester,_1208_1896', 'MN_1433_1896', '44421_1491_1896'],
               'Shawnchester,_1208_1896'),
              (['MN_1433_1896', '44421_1491_1896'], 'MN_1433_1896'),
              (['44421_1491_1896'], '44421_1491_1896'),
              (['Bank_1142_1938', 'Swift_1222_1938', 'Code:_1313_1938', 'LOYD_1402_1938'],
               'Bank_1142_1938'),
              (['Swift_1222_1938', 'Code:_1313_1938', 'LOYD_1402_1938', 'GB_1496_1938'],
               'Swift_1222_1938'),
              (['Code:_1313_1938', 'LOYD_1402_1938', 'GB_1496_1938', '2L_1548_1938'],
               'Code:_1313_1938'),
              (['LOYD_1402_1938', 'GB_1496_1938', '2L_1548_1938'], 'LOYD_1402_1938'),
              (['GB_1496_1938', '2L_1548_1938'], 'GB_1496_1938'),
              (['2L_1548_1938'], '2L_1548_1938'),
              (['IBAN_1377_1980', 'Number:_1460_1980'], 'IBAN_1377_1980'),
              (['Number:_1460_1980'], 'Number:_1460_1980'),
              (['GB22_1115_2021', 'LOYD_1204_2021', '8546_1297_2021', '1463_1382_2021'],
               'GB22_1115_2021'),
              (['LOYD_1204_2021', '8546_1297_2021', '1463_1382_2021', '3082_1464_2021'],
               'LOYD_1204_2021'),
              (['8546_1297_2021', '1463_1382_2021', '3082_1464_2021', '97_1547_2021'],
               '8546_1297_2021'),
              (['1463_1382_2021', '3082_1464_2021', '97_1547_2021'], '1463_1382_2021'),
              (['3082_1464_2021', '97_1547_2021'], '3082_1464_2021'),
              (['97_1547_2021'], '97_1547_2021')]
    print(featureCalculation(ngraml, data[0]))
