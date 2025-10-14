from psychopy import visual, core, event  # import some libraries from PsychoPy
from psychopy.data import MultiStairHandler
from psychopy.hardware import keyboard

myWin = visual.Window([800,600], monitor="testMonitor", units="deg")
text = visual.TextStim(myWin, color='black', text =  "Welcome to the Tutorial.\n \nTo begin, press the right arrow.")
text.draw()
myWin.flip()
event.waitKeys(keyList=['right','num_6'])

#Text
text = visual.TextStim(myWin, color='black', text =  "First, you will see a blank screen with a fixation point.\n \nTo continue, press the right arrow.")
text.draw()
myWin.flip()
event.waitKeys(keyList=['right','num_6'])

#Fixation dot
fixation = visual.GratingStim(win = myWin, color= 'black', colorSpace='rgb', tex=None, mask='circle', size=0.2)
fixation.draw()
myWin.flip()
event.waitKeys(keyList=['right','left','num_4', 'num_6'])

#Text
text = visual.TextStim(myWin, color='black', text =  "In the experiment this fixation point appears briefly. \nDirectly afterwards, the stimulus will appear. \nIt is important to focus on the line in the middle.\n This is called the target.\n Now there will be a red circle around the target but this won't be there in the actual experiment. \nTo continue, press the right arrow.")
text.draw()
myWin.flip()
event.waitKeys(keyList=['right','num_6'])

#Fixation dot and afterwards stimuli with three lines
fixation = visual.GratingStim(win = myWin, color='black', colorSpace='rgb', tex=None, mask='circle', size=0.2)
fixation.draw()
myWin.flip()
core.wait(0.3)

top = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[0,3], sf=0, color=-0.35)
middle = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[0,0], sf=0, color=-0.3)
down = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[0,-3], sf=0, color=-0.35)
red_circle = visual.Circle(win=myWin, fillColor=None, radius = 1.75, lineColor='red')
top.draw()
middle.draw()
down.draw()
red_circle.draw()
myWin.flip()
core.wait(0.2)

#Text
text = visual.TextStim(myWin, color='black', text =  "It is also possible that only the target will appear or more than 3 lines will appear. \nIt is still all about the middle line. \nTherefore stay focused on the target throughout the entire experiment.\n \nTo continue, press the right arrow.")
text.draw()
myWin.flip()
event.waitKeys(keyList=['right','num_6'])

#Fixation dot and afterwards stimulus with only one line
fixation = visual.GratingStim(win = myWin, color= 'black', colorSpace='rgb', tex=None, mask='circle', size=0.2)
fixation.draw()
myWin.flip()
core.wait(0.3)

middle = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[0,0], sf=0, color=-0.3)
middle.draw()
myWin.flip()
core.wait(0.2)

#Text (Time to do it yourself, right=yes, left=no)
text = visual.TextStim(myWin, color='black', text =  "It is time to try it yourself! Focus on the target. \nPress the right arrow if the target is visible. \nPress the left arrow if the target is not visible. \nYou will have 1.2 seconds to react but please answer as quick as possible.\n \nTo continue, press the right arrow.")
text.draw()
myWin.flip()
event.waitKeys(keyList=['right','num_6'])

#3 samples with correction
correct = 0
for trial in range(0,10):
    if correct >= 3:
        break
    else:
        correct = 0 
        #Sample 1 
        fixation = visual.GratingStim(win = myWin, color='black', colorSpace='rgb', tex=None, mask='circle', size=0.2)
        fixation.draw()
        myWin.flip()
        core.wait(0.3)
        
        middle = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[0,0], sf=0, color=-0.3)
        middle.draw()
        myWin.flip()
        core.wait(0.2)
        
        myWin.flip()
        Keys = event.waitKeys(maxWait=1.2, keyList=['left','num_4','right','num_6'])
        
        if Keys: 
            key = Keys[0]
            if 'left'==key or 'num_4'==key:
                text = visual.TextStim(myWin, color='red', text = "Incorrect. \nThe target is visible. Therefore, you need to press the right arrow.\n \nTo continue, press the right arrow.")
                
            elif 'right'==key or 'num_6'==key:
                text = visual.TextStim(myWin, color='green', text = "Correct\n \nTo continue, press the right arrow.")
                correct += 1
            
        else:
            text = visual.TextStim(myWin, color='red', text = "You need to press the left or right arrow.\n \nTo continue, press the right arrow.")
        
        text.draw()
        myWin.flip()
        event.waitKeys(keyList=['right','num_6'])
        
        #Sample 2 
        fixation = visual.GratingStim(win = myWin, color= 'black', colorSpace='rgb', tex=None, mask='circle', size=0.2)
        fixation.draw()
        myWin.flip()
        core.wait(0.3)
        
        top = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[0,3], sf=0, color=-0.35)
        middle = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[0,0], sf=0, color=-0.2)
        down = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[0,-3], sf=0,color=-0.35)
        top.draw()
        middle.draw()
        down.draw()
        myWin.flip()
        core.wait(0.2)
        
        myWin.flip()
        Keys = event.waitKeys(maxWait=1.2, keyList=['left','num_4','right','num_6'])
        
        if Keys: 
            key = Keys[0]
            if 'left'==key or 'num_4'==key:
                text = visual.TextStim(myWin, color='red', text = "Incorrect. \nThe target is visible. Therefore, you need to press the right arrow.\n \nTo continue, press the right arrow.")
                
            elif 'right'==key or 'num_6'==key:
                text = visual.TextStim(myWin, color='green', text = "Correct\n \nTo continue, press the right arrow.")
                correct += 1
            
        else:
            text = visual.TextStim(myWin, color='red', text = "You need to press the left or right arrow.\n \nTo continue, press the right arrow.")
        
        text.draw()
        myWin.flip()
        event.waitKeys(keyList=['right','num_6'])
        
        #Sample 3 
        fixation = visual.GratingStim(win = myWin, color= 'black', colorSpace='rgb', tex=None, mask='circle', size=0.2)
        fixation.draw()
        myWin.flip()
        core.wait(0.3)
        
        top_left = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[-1,3], sf=0, color=-0.2)
        top = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[0,3], sf=0, color=-0.2)
        top_right = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[1,3], sf=0, color=-0.2)
        
        middle_left = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[-1,0], sf=0, color=-0.2)
        middle = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[0,0], sf=0, color=0)
        middle_right = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[1,0], sf=0, color=-0.2)
        
        down_left = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[-1,-3], sf=0, color=-0.2)
        down = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[0,-3], sf=0,color=-0.2)
        down_right = visual.GratingStim(win=myWin, mask = 'sqr', size=[0.1,2], pos=[1,-3], sf=0, color=-0.2)
        
        top_left.draw()
        top.draw()
        top_right.draw()
        middle_left.draw()
        middle.draw()
        middle_right.draw()
        down_left.draw()
        down.draw()
        down_right.draw()
        
        myWin.flip()
        core.wait(0.2)
        
        myWin.flip()
        Keys = event.waitKeys(maxWait=1.2, keyList=['left','num_4','right','num_6'])
        
        if Keys: 
            key = Keys[0]
            if 'left'==key or 'num_4'==key:
                text = visual.TextStim(myWin, color='green', text = "Correct\n \nTo continue, press the right arrow.")
                correct += 1
                
            elif 'right'==key or 'num_6'==key:
                text = visual.TextStim(myWin, color='red', text = "Incorrect. \nThe target is invisible. Therefore you need to press the left arrow.\n \nTo continue, press the right arrow.")
                
            
        else:
            text = visual.TextStim(myWin, color='red', text = "You need to press the left or right arrow.\n \nTo continue, press the right arrow.")
        
        text.draw()
        myWin.flip()
        event.waitKeys(keyList=['right','num_6'])


    

#Text Finish
text = visual.TextStim(myWin, color='black', text =  "Sometimes the target (the line in the middle) will be barely visible and other times there will be no target. \nStill, press right if you see the target and press left if you do not. \nNote that you will not receive feedback during the actual experiment.\n \nTo continue, press the right arrow.")
text.draw()
myWin.flip()
event.waitKeys(keyList=['right','num_6'])
text = visual.TextStim(myWin, color='black', text =  "You have finished the tutorial! \nGood luck with the experiment!")
text.draw()
myWin.flip()
core.wait(5)

myWin.close()
core.quit()