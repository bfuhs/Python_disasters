'''
utilities.py

Some utility functions for CSS 692 project

Matt Snyder
Brendon Fuhs

Usage example:
testDict = reporterDict
filename = "textCSV.csv"
writeDictToCSV(testDict, "textCSV.csv")
'''

import csv


def writeDictToCSV(reporterDict, filename):
    '''
    filename should be string
    reporterDict should be
    key = numerical index
    list is outlet, byline, wordcount, sublist of organizations mentioned in story, possibly other stuff

    '''
    with open(filename, 'wb') as targetFile:
        file_writer = csv.writer( targetFile, delimiter = '\t' ) # or dialect='excel-tab' ?
        for i in range(1, len(reporterDict)+1):
            file_writer.writerow(reporterDict[i])

def readFromCSV(filename):
    '''
    filename is a string
    return ???????????????
    '''
    bigfile = []
    file_reader = csv.reader( open(filename, 'rb') ) ### delimiter?
    for row in file_reader:
        bigfile.append(row)
    return zip(*bigfile) # return transposed
            



