from psychopy.visual import TextStim
from psychopy.event import waitKeys
from psychopy import data, core, visual, event

class Stimulus:
    def __init__(self, win, stimulus_list):
        self.lines = []
        self.win = win

        for stim_def in stimulus_list:
            stim_type = stim_def.get("type", "").lower()
            if stim_type == "line":
                line = visual.Line(
                    win=win,
                    start=stim_def["start"],
                    end=stim_def["end"],
                    pos=stim_def.get("pos", (0, 0)),
                    lineWidth=stim_def.get("lineWidth", 4.2),
                    color=stim_def.get("color", "black"),
                    colorSpace=stim_def.get("colorSpace", "rgb"),
                )
                self.lines.append(line)
            elif stim_type == "text":
                text = TextStim(
                    win=win,
                    text=stim_def["text"],
                    pos=stim_def.get("pos", (0, 0)),
                    color=stim_def.get("color", "black"),
                    colorSpace=stim_def.get("colorSpace", "rgb"),
                    wrapWidth=stim_def.get("wrapWidth", None),
                )
                self.lines.append(text)

        self.fixation = visual.GratingStim(win=win, color=-1, colorSpace='rgb', tex=None, mask='circle', size=0.1)
        self.blank = visual.TextStim(win=win, text="You should not see this", color=win.color, colorSpace='rgb')
        self.diode = visual.GratingStim(win=win, color='black', colorSpace='rgb', tex=None, mask='circle', units='pix', size=80, pos=[-780, -440], autoDraw=True)
        