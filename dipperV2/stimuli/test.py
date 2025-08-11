from psychopy import visual, core

win = visual.Window(fullscr=False,
                       #allowGUI = False, 
                       monitor="testMonitor", 
                       units="deg",
                       colorSpace='rgb',
                       color = [0,0,0],
                       #bpc=(10,10,10),
                       #depthBits=10
                       )
grating = visual.GratingStim(win=win, mask="circle", size=3, pos=[-4,0], sf=3)
fixation = visual.GratingStim(win=win, size=0.5, pos=[0,0], sf=0, rgb=-1)

14#draw the stimuli and update the window
grating.draw()
fixation.draw()
win.update()

#pause, so you get a chance to see it!
core.wait(5.0)