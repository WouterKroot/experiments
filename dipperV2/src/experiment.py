import os
import sys
import threading
from psychopy import data, core, visual, event
import pandas as pd
import pylab
import numpy as np
from typing import Literal

class Experiment:
    def __init__(self, win,
                 subject_id, nTrials, nBlocks, eyeTracker,
                 expConfig, path, nullOdds, myConds):
        self.myWin = win
        self.myConds = myConds
        self.id = subject_id
        self.nTrials = nTrials
        self.nBlocks = nBlocks
        self.expConfig = expConfig
        self.nullOdds = nullOdds
        self.eyeTracker = eyeTracker
        self.path = path
        self.base_name = os.path.basename(path)
        self.stairs = data.MultiStairHandler(stairType='simple',
                                             method='random',
                                             nTrials=self.nTrials,
                                             conditions=self.myConds, originPath=self.path)

    def getBreaks(self):
        totalTrials = int(len(self.myConds) * self.nTrials + self.nullOdds * self.nTrials)
        breakTrials = np.linspace(start=0,stop=totalTrials, num=self.nBlocks,
                                  endpoint=False,dtype=int)[1:]
        return breakTrials, totalTrials
    
    def openDataFile(self):
        os.makedirs(self.path, exist_ok=True)

        fileName = os.path.join(self.path, f"{self.base_name}.csv")
        count = 1

        # Keep incrementing if file exists
        while os.path.exists(fileName):
            fileName = os.path.join(self.path, f"{self.base_name}_{count}.csv")
            count += 1

        self.dataFile = open(fileName, 'w', buffering=1)  # line-buffered
        self.dataFile.write("id,trial,label,FC,TC,response,RT\n")
        
        return fileName

    def run_baseline(self):
        dataFile = self.dataFile
        stairs = self.stairs
        trialClock = core.Clock()
        thisTrial = 0

        for trial, condition in stairs:
            targetIntensity = stairs.currentStaircase.intensity
            thisLabel = condition['label'] 

            stimulus = self.myWin.stimuli[thisLabel]
            for entry in stimulus['components']:
                if entry.get('type') == 'target':
                    entry['line_obj'].contrast = targetIntensity

            # Draw fixation
            self.myWin.win.flip()  # Clear previous frame
            self.myWin.fixation.draw()
            self.myWin.win.flip()
            core.wait(self.myWin.t_fixation)

            # Draw stimulus
            for line in stimulus['draw_lines']:
                line.draw()
            self.myWin.win.flip()
            core.wait(self.myWin.t_stim)

            trialClock.reset()
            self.myWin.drawOrder(self.myWin.blank)
            self.myWin.win.flip()
            
            allKeys = event.waitKeys(maxWait=self.myWin.t_response,
                        keyList=['left','num_4',
                            'right','num_6',
                            'q','escape'])
            
            thisRT = trialClock.getTime()
            if thisRT < self.myWin.t_response:
                core.wait(self.myWin.t_response - thisRT)

            if allKeys:
                for key in allKeys:
                    if key in ['left','num_4']:
                        thisResp = 0
                    elif key in ['right','num_6']:
                        thisResp = 1
                    elif key in ['q','escape']:
                        self.eyeTracker.closeTracker()
                        core.quit()
                    else:
                        raise ValueError(f"Unexpected key: {key}")
            else:
                thisResp = 0
                thisRT = 99

            self.dataFile.write(f"{self.id},{thisTrial},{thisLabel},{000},{stairs.currentStaircase.intensity},{thisResp},{thisRT}\n")
            
            
            if not thisLabel.endswith("_null"): 
                stairs.addResponse(thisResp) # Don't adjust the staircase for a null trial
           
            thisTrial += 1
            
        
        psydat_path = os.path.join(self.path, f"{self.id}_baseline.psydat")
        stairs.saveAsPickle(psydat_path)
        self.dataFile.close()

    def doBreak(self,b):
        m_break = visual.TextStim(self.myWin.win, color='black', height = 32, wrapWidth=600,
                                  text=f"You have finished block {b+1}.\nTime for a break. \nYou can stretch your legs or get some water.\nWait a bit before continuing.")
        m_continue = visual.TextStim(self.myWin.win, color='black', height = 32, wrapWidth=600,
                                  text=f"You have finished block {b+1}.\nYou can continue when ready.\nPress [RIGHT] to continue.\n")
        self.myWin.drawOrder(m_break)
        core.wait(self.myWin.t_break)
        self.myWin.drawOrder(m_continue)
        event.waitKeys(keyList=['right','num_6'])
        self.myWin.countdown()

    def run_main(self,dataFile):
        totalTrials = self.nTrials * self.nBlocks
        
        # we get drawable line objects from the stim_dict in self.myWin.stimuli, then we filter 
        breaks, totalTrials = self.getBreaks()

        stairs = self.stairs

        trialClock = core.Clock()
        thisTrial = 0
        
        self.myWin.countdown()
        for trial, condition in stairs:
            self.myWin.checkQuit()
            
            lines = []
            thisLabel = condition['label'] 
            
            if thisTrial in breaks:
                self.doBreak(b=np.where(breaks == thisTrial)[0][0])

            if self.eyeTracker.doTracking:
                self.eyeTracker.tracker.startRecording(1, 1, 1, 1)

            targetIntensity = stairs.currentStaircase.intensity
            flankerIntensity = stairs.currentStaircase.condition['FC']

            stim_key = condition['stim_key']   # Use stim_key, not label
            stimulus = self.myWin.stimuli[stim_key] 

            for entry in stimulus['components']:
                if entry.get('type') == 'target':
                    entry['line_obj'].contrast = targetIntensity
                else:
                    entry['line_obj'].contrast = flankerIntensity
                lines.append(entry['line_obj'])
            # Random _null chance
            if np.random.random() <= self.nullOdds:
                targetIntensity = 0
                thisLabel += '_null'

            # Draw fixation
            self.myWin.diode.color *= -1 # white -- button on
            self.myWin.drawOrder(self.myWin.fixation)
            core.wait(self.myWin.t_fixation)
            self.blinkDiode() # black -- button off

            # Draw stmiulus
            self.eyeTracker.stimOnset(thisTrial,thisLabel,targetIntensity)
            self.myWin.diode.color *= -1 # white -- button on
            self.myWin.drawOrder(lines)
            core.wait(self.myWin.t_stim)
            self.blinkDiode() # black -- button off

            trialClock.reset()
            self.myWin.drawOrder(self.myWin.blank)
            allKeys = event.waitKeys(maxWait=self.myWin.t_response,
                        keyList=['left','num_4',
                            'right','num_6',
                            'q','escape'])
            
            thisRT = trialClock.getTime()
            if thisRT < self.myWin.t_response:
                core.wait(self.myWin.t_response - thisRT)

            if allKeys:
                for key in allKeys:
                    if key in ['left','num_4']:
                        thisResp = 0
                    elif key in ['right','num_6']:
                        thisResp = 1
                    else:
                        self.eyeTracker.closeTracker()
                        core.quit()

            else:
                thisResp = 0
                thisRT = 99

            self.eyeTracker.logResponse(thisResp,thisRT)
            self.dataFile.write(f"{self.id},{thisTrial},{thisLabel},{condition['FC']},{stairs.currentStaircase.intensity},{thisResp},{thisRT}\n")
            
            if not thisLabel.endswith("_null"): 
                stairs.addResponse(thisResp)

            thisTrial += 1
            if self.eyeTracker.doTracking:
                self.eyeTracker.tracker.stopRecording()
            
            print(thisTrial)
        
            os.makedirs(self.path, exist_ok=True)
            psydat_path = os.path.join(self.path, f"{self.id}_main.psydat")
            stairs.saveAsPickle(psydat_path, fileCollisionMethod='overwrite')
            
        self.myWin.checkQuit()
        self.eyeTracker.closeTracker()
        self.myWin.end()
    
    def getThresholdFromBase(self, file_path):
        threshVal = 0.5
        expectedMin = 0.0

        thisDat = pd.read_csv(file_path)
        thisDat = thisDat[~thisDat['label'].str.endswith('_null')]

        allIntensities = thisDat['TC'].tolist()
        allResponses = thisDat['response'].tolist()

        i, r, n = data.functionFromStaircase(allIntensities, allResponses, bins='unique')
        combinedInten, combinedResp, combinedN = i, r, n
        combinedN = pylab.array(combinedN)

        fit = data.FitLogistic(
            combinedInten, combinedResp,
            expectedMin=expectedMin,
            sems=1.0 / combinedN,
            optimize_kws={'maxfev': int(1e6)}
        )
        thresh = fit.inverse(threshVal)
        print(f'-----------Threshold for [{self.id}, Baseline] is: {thresh}-----------')
        return thresh

        
    def reDoBase(self,thresh):
        m_redo = visual.TextStim(self.myWin.win, color=[1, 1, 1], height = 32, wrapWidth=600,
                                 text = f"Please wait for the experimenter.\nParticipant {self.id} baseline detection threshold:\n{thresh}\nThreshold outside of expected range.\nTry again [y / n]?")
        m_good = visual.TextStim(self.myWin.win, color=[1, 1, 1], height = 32, wrapWidth=600,
                                 text = f"Please wait for the experimenter.\nParticipant {self.id} baseline detection threshold:\n{thresh}\nThreshold inside of expected range.\nGo again [y / n]?")
        if thresh > 0.1 or thresh <= 0:
            self.myWin.drawOrder(m_redo)
            keys = event.waitKeys(keyList=['y','n'])
            if keys:
                for key in keys:
                    if key == 'y':
                        return True
                    else:
                        return False
        else:
            self.myWin.drawOrder(m_good)
            event.waitKeys(keyList=['y','n'])
            return False

    def blinkDiode(self,t=2/60):
        # Defaults to two frames blink (at 60fps)
        # Blinks the diode to indicate the offset of a stimulus
        # Does not draw any new stimuli, flips the window with existing stuff
        self.myWin.diode.color *= -1
        self.myWin.win.flip()
        core.wait(t) # 2 frames
    

        