'''
PyDisasters.py

created by Brendon Fuhs

REQUIRES:
Python 2.7
matplotlib
scipy
powerLaw
numpy
utilities.py
statsmodels
needed to install mpmath module

USAGE:
data = getData()
test = Analysis(data)
test.hazardWindows()
from datetime import datetime
test.chosenStartDate = datetime(1977, 01, 01)
test.powerLawWindows(test.chosenStartDate)
test.analyzeResults()

'''

import numpy as np
import powerlaw as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import math
import itertools as it
import datetime as dt
from statsmodels.tsa.stattools import adfuller
from collections import deque
# from Queue import Queue
from utilities import readFromCSV # reads and writes CSV

# changes integer days to numpy timedelta64 format
def daysToNumpy(days): # probably works eith vectors
    return np.timedelta64(days*10**6*60**2*24)

# changes numpy timedelta64 format to integer days
def numpyToDays(timeDelt):
    return np.asscalar(timeDelt).days

# changes numpy datetime64 format to python's datetime format
def numpyDTtoDT(numpDT): # return numpDT.astype(datetime)
    # http://mail.scipy.org/pipermail/numpy-discussion/2011-December/059386.html
    return dt.datetime.strptime(repr(numpDT), "%Y-%m-%d %H:%M:%S") #.%f

# Types to use:
#### Check for even more case conflicts # Present in new datasets?
# 'Complex Disasters' (not many)
# 'Drought'
# 'Earthquake (seismic activity)'
# 'Epidemic'
# 'Extreme temperature'
# 'Flood'
# 'Industrial Accident' # no merge?
# 'Insect infestation' # MERGE
# 'Insect Infestation'
# 'Mass movement dry' # MERGE
# 'Mass Movement Dry'
# 'Mass movement wet' # MERGE
# 'Mass Movement Wet'
# 'Miscellaneous accident' # no merge?
# 'Storm'
# 'Transport Accident'
# 'Transport accident' # MERGE
# 'Volcano'
# 'Wildfire'
def getData(typeFilter='Earthquake (seismic activity)'): # hard-coded, returns data

    DisasterDates = readFromCSV('DisasterDates.csv')
    colDict = {'type':5, 'day':7, 'month':8, 'year':9} 
    data = {}
    for col in colDict:
        data[col] = DisasterDates[colDict[col]][1:]

    DisasterDetails = readFromCSV('DisasterDetails.csv')

    data['deaths'] = DisasterDetails[3][1:]

    dataLength = len(data['year'])
    for col in data:
        assert len(data[col]) == dataLength
        
    dates = []
    for i in xrange(len(data['year'])):
        yr = str(data['year'][i])
        mo = str(data['month'][i])
        day = str(data['day'][i])
        try:
            dates.append(np.datetime64(yr+'-'+mo+'-'+day))
            ##### Incomplete data will default either to the first or today
        except:
            dates.append(None) # for September 31st and such
    assert len(dates) == dataLength
    data['dates'] = dates

    keepIndices = []
    for i in xrange(len(data['type'])):
        if str(data['type'][i]) == typeFilter:
            keepIndices.append(i)

    filteredData = {}
    for key in data:
        filteredData[key] = [data[key][i] for i in keepIndices]
            
    return filteredData


def AIC(logL, numPars):
    return 2*numPars - 2*logL

def AICc(logL, numPars, numNums):
    return AIC(logL, numPars) + 2*numPars*(numPars+1)/float(numNums - numPars - 1)


class Analysis(object):
    
    def __init__(self, sourceData):
        
        self.widths = [365, 730, 1460, 2920] #### 365 times 1, 2, 4, and 8
        self.stepSizes = [30, 30, 30, 30] ####
        assert len(self.widths) == len(self.stepSizes)

        # The following few lines are a slow and bad way to sort
        sortedDates = []
        sortedDeaths = []
        pairs = sorted(zip(sourceData['dates'], sourceData['deaths']))# np.sort(zip(sourceData['dates'], sourceData['deaths']), axis=0)
        for i in xrange(len(sourceData['dates'])):
            sortedDates.append(pairs[i][0])
            sortedDeaths.append(pairs[i][1])

        #print sortedDeaths[:100]
        #raise Exception('test')

        # eliminate entries without dates OR DEATHS
        toDelete = []
        for i in xrange(len(sortedDates)):
            try:
                sortedDeaths[i] = int(sortedDeaths[i]) ### keep only ones with magnitudes
                if str(type(sortedDates[i])) != "<type 'numpy.datetime64'>":
                    toDelete.append(i)
            except:
                toDelete.append(i)

        #print toDelete ######################
        WHAT = 0 #LOL MODIFY LIST IN LOOP
        for i in toDelete:
            del sortedDates[i+WHAT]
            del sortedDeaths[i+WHAT]
            WHAT -= 1
            # sortedDeaths = np.concatenate(sortedDeaths[:i], sortedDeaths[i+1:]) # because I screwed up np.delete
            
        self.allDates = np.array(sortedDates)
        self.allDeaths = np.array(sortedDeaths)

        self.allOnsetDurations = self.allDates[1:] - self.allDates[:-1]
        self.allOnsetDurations = np.array([ numpyToDays(dat) for dat in self.allOnsetDurations ]) # Vectorization might be better
        self.allOnsetDurations = np.insert(self.allOnsetDurations, 0, 0) # Zero shouldn't matter # np.array([0])

        assert len(self.allDates) == len(self.allDeaths) == len(self.allOnsetDurations)
        print len(self.allDates), "data points"

        #colorSquish = 4 # fiddle with to get colors good
        #colors = cm.rainbow(np.linspace(0, 1, len(self.widths)+colorSquish))[:colorSquish]
        colors = ["black", "blue", "green", "red"]
        self.colorMapping = dict(zip(self.widths, colors)) # colors for plotting # will mess up if lengths don't match
        
        # attributes to be set by methods and stuff
        self.hazardResults = {} # self.hazardResults[windowSize][midDate, various parameters and measurements]
        self.chosenStartDate = None
        self.xmin = None
        self.powerLawResults = {} # self.powerLawResults[windowSize][midDates, various parameters and measurements]
        self.regressionResults = {} # self.regressionResults[windowSize][pvalue, slope, etc]
        
    def runAll(self): ### DON"T USE, set start date manually
        self.hazardWindows()
        self.powerLawWindows(self.chosenStartDate)
        self.analyzeResults()

    def hazardWindows(self):
        self.hazardResults = self.multiWindow(self.hazard, self.allDates, self.allOnsetDurations)
        # self.hazardResults[windowSize][midDate, various parameters and measurements]

        legendInfo = []
        selectedStartDates = []
        plt.figure('hazard rate trends')
        self.hazardResults['1'] = {}
        self.hazardResults['1']['midDates'] = self.allDates ## may be superfluousness around here
        self.hazardResults['1']['DTmidDates'] = unitPlottableMidDates = [ numpyDTtoDT(date) for date in self.allDates ]
        self.hazardResults['1']['ADFuller p-values'] = unitPvals = self.ADFullerWindow(self.allOnsetDurations)
        for date, pval in it.izip(unitPlottableMidDates, unitPvals):
            if pval < 0.01:
                selectedStartDates.append(date)
                break
        for window in self.widths:
            midDates = self.hazardResults[window]['midDates']
            self.hazardResults[window]['DTmidDates'] = plottableMidDates = [ numpyDTtoDT(date) for date in midDates ] # make it faster for further plot cycles
            exPars = np.array(self.hazardResults[window]['exPar'])
            pvals = self.ADFullerWindow(exPars)
            self.hazardResults[window]['ADFuller p-values'] = pvals # listify?
            plt.plot(plottableMidDates, exPars, c=self.colorMapping[window])
            legendInfo.append(str(window)+'-day moving window')
            # Here is where the candidate start date is selected
            for date, pval in it.izip(plottableMidDates, pvals):
                if pval < 0.01:
                     selectedStartDates.append(date)
                     break
        leg=plt.legend(legendInfo, prop={'size':10})
        for i in range(len(self.widths)):
            leg.legendHandles[i].set_linewidth(2.0) # probably a better way to do this
        plt.ylabel('lamda (average onset duration)')
        plt.xlabel('date at center of window')

        plt.figure('data counts per window')
        legendInfo = []
        for window in self.widths:
            legendInfo.append(str(window)+'-day moving window')
            plt.plot(self.hazardResults[window]['DTmidDates'], self.hazardResults[window]['counts'])
        leg=plt.legend(legendInfo, loc='upper left', prop={'size':10})
        for i in range(len(self.widths)):
            leg.legendHandles[i].set_linewidth(2.0)
        plt.ylabel('number of data points in window')
        plt.xlabel('date at center of window')

        self.chosenStartDate = selectedStartDates[0]
        for date in selectedStartDates:
            if date > self.chosenStartDate:
                self.chosenStartDate = date
        print selectedStartDates ## Describe which is which
        print self.chosenStartDate

        legendInfo = []
        plt.figure('ADFuller stationarity tests on average onset durations in moving windows')
        for window in self.hazardResults:
            legendInfo.append(str(window)+'-day moving window')
            # plot p-values
            if window == '1':
                color = 'orange'
            else:
                color = self.colorMapping[window]
            plt.plot(self.hazardResults[window]['DTmidDates'], self.hazardResults[window]['ADFuller p-values'], c=color)
        leg=plt.legend(legendInfo, loc='upper left', prop={'size':10})
        for i in range(len(self.hazardResults)): 
            leg.legendHandles[i].set_linewidth(2.0)
        plt.ylabel('ADF test p-value')
        plt.xlabel('start date of ADF test')
        
        # plot other stuff too
        plt.show()

    #### THIS ONE IS TOO SLOW FOR MY LIKING !!!########
    # values should be slice-able, like numpy maybe
    def ADFullerWindow(self, values): # returns startDate and vector of p-values

        #brokenFlatnessYet = False
        i = 50 # number of values to skip
        pvals = deque([None]*i)
        numDates = len(values)
        startIndex = 0
        while i < numDates:# - 1: 
            if i%100 == 0:
                print i
            i += 1
            dataSubset = values[-i:]
            pval = self.stationarity(dataSubset)
            pvals.appendleft(pval)
            #if pval > 0.05 and brokenFlatnessYet == False:
            #    startIndex = numDates - (i - 1) # selects last one checked
            #    brokenFlatnessYet = True

        #assert len(pvals) == len(dates)
        return pvals # numpyDTtoDT(dates[startIndex]), 

    def stationarity(self, timeSeries): ### MAYBE MODIFY SO IT RETURNS TRUE FOR LOW REGRESSION RESULTS ???
        # http://statsmodels.sourceforge.net/stable/generated/statsmodels.tsa.stattools.adfuller.html#statsmodels.tsa.stattools.adfuller
        # Null assumes unit root (non-stationary)
        timeSeries = np.array([x for x in timeSeries if (not np.isinf(x) and not np.isnan(x))])
        ADFtest = adfuller(timeSeries) # Can't feed this NaN or inf #
        return ADFtest[1]# '1' slot is pvalue

    def powerLawWindows(self, startDate):
        
        startIndex = 0
        for i in xrange(len(self.allDates)):
            if numpyDTtoDT(self.allDates[i]) > startDate:
                break
            startIndex += 1
        dates = self.allDates[i:]
        deaths = self.allDeaths[i:]

        print len(deaths), 'data points'

        aggregateResult = self.powerLaw(deaths, showAndTest=True) # sets xmin
        print aggregateResult
        
        self.powerLawResults = self.multiWindow(self.powerLaw, dates, deaths)
        # self.powerLawResults[windowSize][midDates, various parameters and measurements]

    def analyzeResults(self):
        self.regressionResults = {} # to be set herein
        self.regressionResults['goodStat'] = {}
        self.regressionResults['alpha'] = {}
        plt.figure('change in alpha')
        legendInfo = []
        for windowWidth in self.powerLawResults: # or in self.widths
            self.regressionResults['alpha'][windowWidth] = {}
            midDates = self.powerLawResults[windowWidth]['midDates']
            plottableDates = [ numpyDTtoDT(date) for date in midDates ]
            mathableDates = [date.toordinal() - 693596 for date in plottableDates ]
            alphas = self.powerLawResults[windowWidth]['alpha']
            self.regressionResults['alpha'][windowWidth] = stats.linregress(mathableDates, alphas) ## Sometimes error
            plt.plot(plottableDates, alphas, c=self.colorMapping[windowWidth])
            legendInfo.append(str(windowWidth)+'-day moving window')
            intercept, slope = self.regressionResults['alpha'][windowWidth][1] , self.regressionResults['alpha'][windowWidth][0]
            yLinePoints = [ intercept + date*slope for date in mathableDates ]
            plt.plot(plottableDates, yLinePoints, c=self.colorMapping[windowWidth], linestyle='--')
            legendInfo.append("fit line") # +"with slope ="+str(slope)
        
        leg = plt.legend(legendInfo, prop={'size':10})
        for i in range(len(self.powerLawResults)*2):
            leg.legendHandles[i].set_linewidth(2.0)
        plt.ylabel('alpha parameter')
        plt.xlabel('date at center of window')

        print self.regressionResults['alpha'] # would have been nice to do in table form

        plt.figure('change in K-S statistic')
        for windowWidth in self.powerLawResults:
            self.regressionResults['goodStat'][windowWidth] = {}
            midDates = self.powerLawResults[windowWidth]['midDates']
            plottableDates = [ numpyDTtoDT(date) for date in midDates ]
            mathableDates = [date.toordinal() - 693596 for date in plottableDates ]
            goodStats = self.powerLawResults[windowWidth]['goodStat']
            self.regressionResults['goodStat'][windowWidth] = stats.linregress(mathableDates, goodStats) ## Sometimes error
            plt.plot(plottableDates, goodStats, c=self.colorMapping[windowWidth])
            legendInfo.append(str(windowWidth)+'-day moving window')
            intercept, slope = self.regressionResults['goodStat'][windowWidth][1] , self.regressionResults['goodStat'][windowWidth][0]
            yLinePoints = [ intercept + date*slope for date in mathableDates ]
            plt.plot(plottableDates, yLinePoints, c=self.colorMapping[windowWidth], linestyle='--')
            legendInfo.append("fit line")
        leg = plt.legend(legendInfo, prop={'size':10})
        for i in range(len(self.powerLawResults)*2):
            leg.legendHandles[i].set_linewidth(2.0) 
        plt.ylabel('K-S statistic')
        plt.xlabel('date at center of window')
            
        print self.regressionResults['goodStat']
        plt.show()

    def multiWindow(self, analysisFunction, Xdates, Ydata): # analysisType is 'powerLaw', 'hazard'

        multiResults = {} # dict of windowWidth:paramResultsDict
        
        for width, step in it.izip(self.widths, self.stepSizes):
            theseResults = self.movingWindow(analysisFunction, width, step, Xdates, Ydata)
            multiResults[width] = theseResults

        return multiResults # dict of windowWidth:paramResultsDict
    
    def movingWindow(self, analysisFun, width, step, Xdates, Ydata):

        results = {}
        count = 0
        reportFreq = 100
        for midDate, subset in self.windows(Xdates, Ydata, daysToNumpy(width), daysToNumpy(step)):
            if count % reportFreq == 0:
                print count
                
            result = analysisFun(subset)
            
            if count == 0:
                for key in result:
                    results[key] = deque([result[key]])
                results['midDates'] = deque([midDate])
                results['counts'] = deque([len(subset)])
            else:
                for key in result:
                    results[key].append(result[key])
                results['midDates'].append(midDate)
                results['counts'].append(len(subset))
            
            count += 1

        testLength = len(results['counts'])
        for key in results:
            assert len(results[key]) == testLength

        return results 


    # Yields window boundaries and subsetted data
    ### At this point, framekeys are assumed to be dates ### Probably a better way to do this
    def windows(self, frameKeys, Ydata, width, step):
        # width and step should be in a format compatible with frameKeys
        # data should be numpy
        
        windowSubset = np.array([]) # Queue() # deque() # 
        windowFrameKeys = np.array([]) # Queue() # deque() # 
        # will progressively pop stuff off of frameKeys and data and add them to the window stuff

        firstRound = True
        for beginWindow, endWindow in self.windowFrames(frameKeys, width, step):

            numToDrop = numToAdd = 0

            # determines number of entries to drop this frame
            for frameKey, datum in it.izip(windowFrameKeys, windowSubset):
                if frameKey > beginWindow:
                    break
                numToDrop += 1

            # determines number of entries to add this frame
            for frameKey, datum in it.izip(frameKeys, Ydata):
                if frameKey > endWindow:
                    break
                numToAdd += 1

            # change windows and future data
            windowSubset = windowSubset[numToDrop:]
            windowFrameKeys = windowFrameKeys[numToDrop:]
            if firstRound == True:
                windowSubset = Ydata[:numToAdd]
                windowFrameKeys = frameKeys[:numToAdd]
            else:
                windowSubset = np.append(windowSubset, Ydata[:numToAdd])
                windowFrameKeys = np.append(windowFrameKeys, frameKeys[:numToAdd])
            Ydata = Ydata[numToAdd:]#################### HOPE THIS DOESN'T CHANGE SOMETHING ELSE ###########
            frameKeys = frameKeys[numToAdd:]
            midDate = beginWindow + daysToNumpy( numpyToDays(endWindow - beginWindow) / 2)
            yield midDate, windowSubset
            firstRound = False

    # Tracks the window frame
    def windowFrames(self, frameKeys, width, step):
        # width and step must be of types compatible with framekeys
        beginWindow = frameKeys[0]
        endWindow = beginWindow + width
        while True:
            if endWindow > frameKeys[-1]:
                raise StopIteration
            yield beginWindow, endWindow
            beginWindow += step
            endWindow += step
    
    def powerLaw(self, dataSubset, showAndTest=False): # fixing xmin this time
        ############ MAKE SURE XMIN IS HAPPENING RIGHT

        count = len(dataSubset) ############# POSSIBLY CHECK IF THIS IS TOO SMALL

        problem = False
        if self.xmin == None:
            try: results = pl.Fit(dataSubset)
            except: problem = True
        else:
            try: results = pl.Fit(dataSubset, xmin=self.xmin)
            except: problem = True
            
        if problem:
            xmin = None #### Not sure exactly what I should be returning here
            alpha = None ### Or should there error catching be happening at a higher level?
            goodStat = None
            portionUsable = None
        else:
            xmin = results.power_law.xmin
            alpha = results.power_law.alpha
            goodStat = results.power_law.D
            #portionUsable = usableCount / float(totalCount)

        if showAndTest == False:
            return {'xmin':xmin,
                    'alpha':alpha,
                    'goodStat':goodStat } #,
                    #'portionUsable':portionUsable}


        if self.xmin == None:
            stuff = pl.find_xmin(dataSubset)
            print stuff #################################################################3
            self.xmin = stuff[0]
        fit = pl.Fit(dataSubset, xmin=self.xmin)
        fit.plot_ccdf()
        fit.power_law.plot_ccdf()
        fit.lognormal.plot_ccdf()
        #fit.exponential.plot_ccdf() # way off
        fit.truncated_power_law.plot_ccdf()
        #fit.stretched_exponential.plot_ccdf() ### THIS DOESN"T PLOT
        # This is only showing three lines - maybe there's some overlap?

        leg = plt.legend(["empirical", "power law", "log-normal", "truncated power law"], loc='lower left', prop={'size':10}) # , "stretched exponential"
        for i in range(4): # number of stuff on plot
            leg.legendHandles[i].set_linewidth(2.0) # probably a better way to do this

        pLawLogL = sum(fit.power_law.loglikelihoods(dataSubset))
        logNormLogL = sum(fit.lognormal.loglikelihoods(dataSubset))
        truncPLawLogL = sum(fit.truncated_power_law.loglikelihoods(dataSubset))
        strchExpLogL = sum(fit.stretched_exponential.loglikelihoods(dataSubset))
        plt.ylabel('CCDF(number of deaths)')
        plt.xlabel('number of deaths')
        
        #####################
        # print fit.fixed_xmin#####################
        numNums = len(dataSubset)
        AICtable = {'power law': {'AIC': AIC(pLawLogL, 1), 'AICc': AICc(pLawLogL, 1, numNums)},
                    'log-normal': {'AIC': AIC(logNormLogL, 2), 'AICc': AICc(logNormLogL, 2, numNums)},
                    'truncated power law': {'AIC': AIC(truncPLawLogL, 2), 'AICc': AICc(truncPLawLogL, 2, numNums)},
                    'stretched exponential': {'AIC': AIC(strchExpLogL, 2), 'AICc': AICc(strchExpLogL, 2, numNums)} }

        plt.show()

        print AICtable
        self.xmin = xmin
        return AICtable
        # impose xmax?


    ###### commented-out stuff either superfluous or not working properly
    def hazard(self, dataSubset): # data should be numpy vector of onset durations
        
        ### exLoc, exLam = stats.expon.fit(dataSubset, floc=0) # exLam is the parameter here # THIS JUST AVERAGE RIGJHT?
        exLam = dataSubset.sum() / float(len(dataSubset)) ################ NEED TO ELIMINATE Nones/NaNs????
        #weibEx, weibShape, weibLoc, weibScale = stats.exponweib.fit(dataSubset, floc=0, f0=1) # weibShape and weibScale are params here

        #expLogLarray = stats.expon.logpdf(dataSubset, exLam) ## I think these are already numpy vectorized
        #expLogL = expLogLarray.sum()
        #weibLogLarray = stats.exponweib.logpdf(dataSubset, 1, weibShape, scale=weibScale)
        #weibLogL = weibLogLarray.sum()

        #D = -2*expLogL + 2*weibLogL ################### WASN'T WORKING BEFORE
        #alpha = .01 # I should probably think about multiple hypothesis testing or something
        #df=1
        ### chi square with 1 df

        #rejectExp = False
        #if D > stats.chi2.ppf(alpha, df): # args can't be keyworded it seems # What test am I doing here? LRT?? $$$$$##############
        #    rejectExp = True
        
        return {'exPar':exLam }#,
                #'weibShape':weibShape,
                #'weibScale':weibScale,
                #'goodstat':D,
                #'rejectExp':rejectExp}


#################################### Get rid of this stuff ################# 
'''
    def showResults(self): # This won't be the most efficient, but Oh Well

        labelDict = { 'powerLaw': ['midDate', 'counts', 'xmin', 'alpha', 'goodStat'],
                      'hazard': ['midDate', 'counts', 'expPar', 'weibShape', 'weibScale', 'goodstat', 'rejectExp'] }
        labels = labelDict[self.analysisType]

        plottableDates = [[numpyDTtoDT(nDate) for nDate in self.multiResults[j][labels[0]]] for j in xrange(len(self.widths))]

        for i in xrange(1,len(labels)):
            
            plt.figure(i)
            for j in xrange(len(self.widths)):
                plt.plot(plottableDates[j], self.multiResults[j][labels[i]])
            plt.legend([str(width)+'-day moving window' for width in self.widths], loc='upper left')
            plt.xlabel('Date at center of window')
            plt.ylabel(labels[i])

        plt.show()
'''
