import pandas as pd
from utils.helperFunctions import CONFIG_PATH, getConfig

if __name__ == '__main__':
    countries = pd.read_csv(getConfig("pathToCountryGazetteer", CONFIG_PATH), header=None)
    countries.columns = ["Country"]
    countries.to_csv(getConfig("pathToCountryGazetteer", CONFIG_PATH))

    cities = pd.read_csv(getConfig("pathToCityGazetteer", CONFIG_PATH), header=None, sep="\t")
    cities.columns = ['geonameid',
                      'name',
                      'asciiname',
                      'alternatenames',
                      'latitude',
                      'longitude',
                      'feature class',
                      'feature code',
                      'country code',
                      'cc2',
                      'admin1 code',
                      'admin2 code',
                      'admin3 code',
                      'admin4 code',
                      'population',
                      'elevation',
                      'dem',
                      'timezone',
                      'modification date']
    cities.to_csv(getConfig("pathToCityGazetteer", CONFIG_PATH))

    # Does not contain numeric variants of ZIP Codes - only the string variant
    zipCodes = pd.read_csv(getConfig("pathToZIPGazetteer", CONFIG_PATH), header=None, sep="\t", low_memory=True)
    zipCodes.columns = ['geonameid',
                        'name',
                        'asciiname',
                        'alternatenames',
                        'latitude',
                        'longitude',
                        'feature class',
                        'feature code',
                        'country code',
                        'cc2',
                        'admin1 code',
                        'admin2 code',
                        'admin3 code',
                        'admin4 code',
                        'population',
                        'elevation',
                        'dem',
                        'timezone',
                        'modification date']
    zipCodes.to_csv(getConfig("pathToZIPGazetteer", CONFIG_PATH))
