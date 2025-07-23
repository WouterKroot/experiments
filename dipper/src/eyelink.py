import os
from psychopy.visual import TextStim
from psychopy.event import waitKeys
from psychopy import data, core, visual, event
import pylink

class EyeTracker:
    def __init__(self,id,
                 ip = "100.1.1.2:255.255.255.0",
                 doTracking = True):
        self.id = id
        self.ip = ip
        self.doTracking = doTracking
        self.eyeHostFile = str(self.id)+'.edf'
        self.eyeLocalFile = "EyeLink/"+self.eyeHostFile

    def startTracker(self):
        if not self.doTracking:
            return
        self.tracker = pylink.EyeLink(self.ip)
        self.tracker.openDataFile(self.eyeHostFile)
        self.tracker.sendCommand("screen_pixel_coords = 0 0 1919 1079")
        pylink.openGraphics()
        self.tracker.doTrackerSetup()
        pylink.closeGraphics()

    def closeTracker(self):
        if not self.doTracking:
            return
        self.tracker.closeDataFile()
        self.tracker.receiveDataFile(self.eyeHostFile, self.eyeLocalFile) # Takes closed data file from 'src' on host PC and copies it to 'dest' at Stimulus PC
        self.tracker.close()

    def stimOnset(self,trial_id,condition,contrast):
        if not self.doTracking:
            return
        self.tracker.sendMessage(f"TRIAL_START {trial_id} CONTRAST {contrast} CONDITION {condition}")

    def logResponse(self,response,rt):
        if not self.doTracking:
            return
        self.tracker.sendMessage(f"RESPONSE {response} RT {rt}")

