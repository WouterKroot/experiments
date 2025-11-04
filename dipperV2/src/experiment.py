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
        totalTrials = int(len(self.myConds) * self.nTrials * (1 + self.nullOdds))
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
        print("Baseline done, break!")

    def doBreak(self,b, middle=False):
        print(f'this is middle: {middle}')
        
        if middle:
            m_break = visual.TextStim(self.myWin.win, color='black', height = 32, wrapWidth=600,
                        text= (
                f"You have finished block {b+1}.\n"
                "Take a LARGE break (~10 minutes).\n"
                "Stretch your legs, get some water, or rest your eyes.\n"
                "Press [RIGHT] when ready to continue."
            ))

            self.myWin.drawOrder(m_break)
            event.waitKeys(keyList=['right', 'num_6'])
            self.myWin.countdown()
            
        else:
            m_break = visual.TextStim(self.myWin.win, color='black', height = 32, wrapWidth=600,
                                    text=(f"You have finished block {b+1}.\nTime for a break. \nYou can stretch your legs or get some water.\nWait a bit before continuing.\n"
                                    "Press [RIGHT] to continue."))

            self.myWin.drawOrder(m_break)
            event.waitKeys(keyList=['right','num_6'])
            self.myWin.countdown()
        
    def run_tutorial(self):
        win = self.myWin.win  

        def show_text(msg, wait_keys=['right', 'num_6'], color='black'):
            text = visual.TextStim(win, color=color, text=msg)
            text.draw(); win.flip()
            event.waitKeys(keyList=wait_keys)

        show_text("Welcome to the Tutorial.\n\nTo begin, press the right arrow.")

        show_text("First, you will see a blank screen with a fixation point.\n\nTo continue, press the right arrow.")
        
        self.myWin.drawOrder(self.myWin.fixation)
        event.waitKeys(keyList=['right', 'left', 'num_4', 'num_6'])

        show_text(
            "In the experiment this fixation point appears briefly.\n"
            "Directly afterwards, the stimulus will appear.\n"
            "It is important to focus on the line in the middle (the target).\n"
            "A red circle will mark it here, but not in the real experiment.\n\n"
            "To continue, press the right arrow."
        )

        top = visual.line.Line(win=self.myWin.win, start=(0, 30), end=(0, 70), pos=(0, 60), ori=0.0, contrast=1.0, color=-1)
        middle = visual.line.Line(win=self.myWin.win, start=(0, -20), end=(0, 20), pos=(0, 0), ori=0.0, contrast=1.0, color=-1)
        bottom = visual.line.Line(win=self.myWin.win, start=(0, -70), end=(0, -30), pos=(0, -60), ori=0.0, contrast=1.0, color=-1)
        red_circle = visual.Circle(win=self.myWin.win, fillColor=None, radius=35, lineColor='red', lineWidth=3)

        self.myWin.drawOrder(self.myWin.fixation)
        core.wait(self.myWin.t_fixation)
        
        self.myWin.drawOrder([bottom, middle, top, red_circle])
        event.waitKeys(keyList=['right', 'left', 'num_4', 'num_6'])

        show_text(
            "Now it's your turn!\n"
            "Focus on the target.\n"
            "Press RIGHT if the target is visible.\n"
            "Press LEFT if it is not visible.\n"
            "You have 1.2 seconds to respond.\n\n"
            "To continue, press the right arrow."
        )
        
        def show_trial(stims, visible=True):
                """Helper: show fixation, then stimuli, then collect response."""
                # Fixation
                self.myWin.drawOrder(self.myWin.fixation)
                core.wait(0.3)

                # Stimuli
                self.myWin.drawOrder(stims)
                core.wait(0.2)
                self.myWin.win.flip()  # blank screen

                # Response
                keys = event.waitKeys(
                    maxWait=1.2,
                    keyList=['left', 'num_4', 'right', 'num_6']
                )

                if not keys:
                    show_text("You need to press LEFT or RIGHT.\n\nPress RIGHT to continue.",
                            ['right', 'num_6'], 'red')
                    return 0

                key = keys[0]
                if (visible and key in ['right', 'num_6']) or (not visible and key in ['left', 'num_4']):
                    show_text("Correct!\n\nPress RIGHT to continue.", ['right', 'num_6'], 'green')
                    return 1
                else:
                    msg = ("Incorrect.\n" +
                        ("Target visible → press RIGHT." if visible else "Target invisible → press LEFT.") +
                        "\n\nPress RIGHT to continue.")
                    show_text(msg, ['right', 'num_6'], 'red')
                    return 0
        
        correct_streak = 0
        trial_num = 0

        while correct_streak < 3 and trial_num < 10:
            trial_num += 1
            correct_streak = 0  # restart streak each round

            # Trial 1: visible (single line)
            middle = visual.Line(
                win=self.myWin.win, start=(0, -20), end=(0, 20),
                pos=(0, 0), ori=0.0, color=-1, lineWidth=3
            )
            correct_streak += show_trial([middle], visible=True)

            # Trial 2: visible (3 lines)
            top = visual.Line(
                win=self.myWin.win, start=(0, -20), end=(0, 20),
                pos=(0, 60), ori=0.0, color=-1, lineWidth=3
            )
            middle = visual.Line(
                win=self.myWin.win, start=(0, -20), end=(0, 20),
                pos=(0, 0), ori=0.0, color=-1, lineWidth=3
            )
            bottom = visual.Line(
                win=self.myWin.win, start=(0, -20), end=(0, 20),
                pos=(0, -60), ori=0.0, color=-1, lineWidth=3
            )
            correct_streak += show_trial([bottom, middle, top], visible=True)

            # Trial 3: invisible (9 lines grid)
            all_stims = []
            for y in [60, 0, -60]:
                for x in [-100, 0, 100]:
                    # Skip the center (target) completely
                    if x == 0 and y == 0:
                        continue
                    all_stims.append(
                        visual.Line(
                            win=self.myWin.win, start=(0, -20), end=(0, 20),
                            pos=(x, y), ori=0.0, color=-1, lineWidth=3
                        )
                    )
            correct_streak += show_trial(all_stims, visible=False)

        show_text(
            "Sometimes the target will be barely visible, or absent.\n"
            "Press RIGHT if you see it, LEFT if you do not.\n"
            "You won't receive feedback during the real experiment.\n\n"
            "To continue, press the right arrow."
        )

        show_text("You have finished the tutorial!\nGood luck with the experiment!\n"
                  "Press RIGHT to continue to the actual experiment.")
        core.wait(3)
        event.waitKeys(keyList=['right', 'left', 'num_4', 'num_6'])
        
    def run_main(self, dataFile): 
        breaks, totalTrials = self.getBreaks() #total trials with null trials for correct breaks
        stairs = self.stairs
        totalStaircaseTrials = int(len(self.myConds) * stairs.nTrials) # staircase trials
    
        middle_index = len(breaks) // 2
        middle_trial = breaks[middle_index] if len(breaks) > 0 else -1

        print(f"Total trials with null: {totalTrials}, Breaks at trials: {breaks}, middle index: {middle_index}, middle trial: {middle_trial}")

        trialClock = core.Clock()
        thisTrial = 0         # counts all displayed trials (including nulls)
        stairTrialCount = 0   # counts only trials added to staircase

        self.myWin.countdown()

        # Loop until all staircase trials are completed
        while stairTrialCount < totalStaircaseTrials: 
            self.myWin.checkQuit()
            print('===============')
            #print(f"Total trials with null: {stairs.totalTrials},\n total staircase trials (no null): {totalStaircaseTrials}, total for breaks: {totalTrials}")
            
            print(f"Total trials with null: {totalTrials}, Breaks at trials: {breaks}, middle index: {middle_index}, middle trial: {middle_trial}")
            #print(f"Total trials from the staircase: {stairs.totalTrials}")
            print(f"Total trials no null, totalStaircaseTrials: {totalStaircaseTrials}")
            print(f"Trial: {thisTrial}, staircase count: {stairTrialCount}")
            
            # --- Random null trial ---
            isNull = np.random.random() <= self.nullOdds
            if isNull:
                print("Null trial")
                currentStair = stairs.currentStaircase
                condition = currentStair.condition
                thisLabel = condition['label']
                thisLabel += '_null'
                targetIntensity = 0
            else:
                stairs.next()  
                currentStair = stairs.currentStaircase
                condition = currentStair.condition
                thisLabel = condition['label']
                targetIntensity = currentStair.intensity

            # --- Handle breaks ---
            if thisTrial in breaks:
                b_idx = np.where(breaks == thisTrial)[0][0]
                middle_break = (thisTrial == middle_trial)
                self.doBreak(b=b_idx, middle=middle_break)
                # --- Save staircase periodically ---
                os.makedirs(self.path, exist_ok=True)
                psydat_path = os.path.join(self.path, f"{self.id}_main.psydat")
                stairs.saveAsPickle(psydat_path, fileCollisionMethod='overwrite')

            # --- Eye tracker start ---
            if self.eyeTracker.doTracking:
                self.eyeTracker.tracker.startRecording(1, 1, 1, 1)

            # --- Prepare stimulus ---
            lines = []
            flankerIntensity = currentStair.condition['FC']
            stim_key = condition['stim_key']
            stimulus = self.myWin.stimuli[stim_key]

            for entry in stimulus['components']:
                if entry.get('type') == 'target':
                    entry['line_obj'].contrast = targetIntensity
                else:
                    entry['line_obj'].contrast = flankerIntensity
                lines.append(entry['line_obj'])

            # --- Draw fixation ---
            self.myWin.diode.color *= -1
            self.myWin.drawOrder(self.myWin.fixation)
            core.wait(self.myWin.t_fixation)
            self.blinkDiode()

            # --- Draw stimulus ---
            self.eyeTracker.stimOnset(thisTrial, thisLabel, targetIntensity)
            self.myWin.diode.color *= -1
            self.myWin.drawOrder(lines)
            core.wait(self.myWin.t_stim)
            self.blinkDiode()

            # --- Collect response ---
            trialClock.reset()
            self.myWin.drawOrder(self.myWin.blank)
            allKeys = event.waitKeys(maxWait=self.myWin.t_response,
                                    keyList=['left','num_4','right','num_6','q','escape'])
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

            # --- Log response ---
            self.eyeTracker.logResponse(thisResp, thisRT)
            self.dataFile.write(f"{self.id},{thisTrial},{thisLabel},{condition['FC']},{currentStair.intensity},{thisResp},{thisRT}\n")

            # --- Add response only if not null ---
            if not isNull:
                stairs.addResponse(thisResp)
                stairTrialCount += 1

            # Increment total trial counter for breaks / logging
            thisTrial += 1

            if self.eyeTracker.doTracking:
                self.eyeTracker.tracker.stopRecording()

        # --- End of experiment ---
        self.myWin.checkQuit()
        self.myWin.end()
        self.eyeTracker.closeTracker()
    
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
    

        