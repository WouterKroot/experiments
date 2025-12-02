from psychopy import core, visual, event
import math
import sys

class Window:
    def __init__(self, window, expConfig):
        self.win = window
        self.expConfig = expConfig
        self.fixation = visual.GratingStim(win = window, color= [1,1,1], colorSpace='rgb',     tex=None, mask='circle', size=5)
        self.blank = visual.TextStim(win = window, text="You should not see this", color=window.color, colorSpace='rgb')
        self.diode = visual.GratingStim(win = window, color=[1, 1, 1],colorSpace='rgb',tex=None,mask='circle',units='pix',size=80,pos=[-780,-440],autoDraw=True)

        self.t_stim = expConfig['fixed_params']['t_stim']
        self.t_fixation = expConfig['fixed_params']['t_fixation']
        self.t_response = expConfig['fixed_params']['t_response']
        self.t_break = expConfig['fixed_params']['t_break']
              
    def checkQuit(self):
        if 'escape' in event.getKeys():
            self.win.close()
            core.quit()
        
    def drawOrder(self, stimuli):
        if not isinstance(stimuli, (list,tuple)):
            stimuli.draw()
        elif isinstance(stimuli, (list,tuple)):
            for stimulus in stimuli:
                stimulus.draw()
        else:
            raise ValueError("Impossible instance of `stimuli` argument.")
        self.win.flip()
        return

    def countdown(self, duration=3):
        t = core.CountdownTimer(int(duration))
        while t.getTime() >= 0:
            cd = visual.TextStim(self.win, color = 'black', height = 32, text = f"Experiment will continue in {math.ceil(t.getTime())}...")
            self.drawOrder(cd)
            
    def intro_baseline(self):
        intro_msg = visual.TextStim(self.win, color='black', height = 32, wrapWidth=600,
                                  text = "Welcome to the baseline.\nYou will see a white dot that is quickly replaced by verticle line.\n\nThe line will only appear briefly.\nAfter it disappears, please respond to indicate whether you saw the line. \n\nPress [LEFT] for NO, or press [RIGHT] for YES.\nYou should respond as quickly as possible.\nPlease always look at the center of the screen.\nPress [RIGHT] when you are ready to begin.")
        self.drawOrder(intro_msg)
        event.waitKeys(keyList=['right','num_6'])
        self.countdown()
        
    def intro_experiment(self):
        intro_msg = visual.TextStim(self.win, color='black', height = 32, wrapWidth=600,
                                  text = "Welcome to the REAL experiment. \nThe line will only appear briefly.\nAfter it disappears, please respond to indicate whether you saw the middle line.\nPress [LEFT] for NO, or press [RIGHT] for YES.\nYou should respond as quickly as possible.\nPlease always look at the center of the screen.\nPress [RIGHT] when you are ready to begin.")
        self.drawOrder(intro_msg)
        event.waitKeys(keyList=['right','num_6'])
        self.countdown()
        
    def midway(self,nBlocks):
        midway_msg = visual.TextStim(self.win, color= 'black', height = 32, wrapWidth=600,
                                   text = f"The next portion will be the first of {nBlocks} blocks.\nYou will now also see additional lines.\nThere will sometimes be a line ABOVE and a line BELOW the target.\nYou should only respond based on the line in the MIDDLE.\nThe procedure will be the same.\nPress [RIGHT] to continue.")
        self.drawOrder(midway_msg)
        event.waitKeys(keyList=['right','num_6'])
        self.countdown()
    
    def end(self):
        m_end = visual.TextStim(self.win, color=[1, 1, 1], height = 32, wrapWidth=600,
                                text = "Thanks for Participating!\nIt's finally over!\n:)""")
        self.drawOrder(m_end)
        core.wait(5)
    

        