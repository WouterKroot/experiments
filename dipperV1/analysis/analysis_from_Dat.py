#%%
from psychopy import data, gui, core
from psychopy.tools.filetools import fromFile
import pylab
#Open a dialog box to select files from
files = gui.fileOpenDlg('.')
if not files:
    core.quit()

#get the data from all the files
allIntensities, allResponses = [],[]
for thisFileName in files:
    # thisDat = fromFile(thisFileName)
    # allIntensities.append( thisDat.intensity )
    # allResponses.append( thisDat.data )
    
    thisDat = fromFile(thisFileName)

    # Check if it's a MultiStairHandler
    if hasattr(thisDat, 'staircases'):
        for stair in thisDat.staircases:
            allIntensities.append(stair.intensities)
            allResponses.append(stair.data)
    else:  # single StairHandler
        allIntensities.append(thisDat.intensities)
        allResponses.append(thisDat.data)

    print('Succes!')

# %%
thisDat = fromFile('/Users/wouter/Documents/phd/projects/psychophysics/experiments/dipperV2/Output/Exp/Main/130_main/130_main.psydat')

allIntensities, allResponses = [],[]
threshVal = 0.8
expectedMin = 0.0

for staircase in thisDat.staircases:
    intensities = (staircase.intensities)
    responses = (staircase.data)
    
    combinedInten, combinedResp, combinedN = data.functionFromStaircase(intensities, responses, bins = 10) #bins='unique')
    
    combinedN = pylab.array(combinedN)

    fit = data.FitLogistic(
        combinedInten, combinedResp,
        expectedMin=expectedMin,
        sems=1.0 / combinedN,
    )
    thresh = fit.inverse(threshVal)
    
    smoothInt = pylab.arange(min(combinedInten), max(combinedInten), 0.001)
    smoothResp = fit.eval(smoothInt)
    
    pylab.subplot(122)
    pylab.plot(smoothInt, smoothResp, '-')
    pylab.plot([thresh, thresh],[0,0.8],'--'); pylab.plot([0, thresh],\
    [0.8,0.8],'--')
    pylab.title('threshold = %0.3f' %(thresh))
    #plot points
    pylab.plot(combinedInten, combinedResp, 'o')
    pylab.show()
   
    

# allIntensities, allResponses = [],[]
# for staircase in thisDat.staircases:
#     intensities = (staircase.intensities)
#     responses = (staircase.data)
    
#     combinedInten, combinedResp, combinedN = data.functionFromStaircase(intensities, responses, 20)
#     fit = data.FitWeibull(combinedInten, combinedResp) #, guess=[0.001, 0.1])
#     smoothInt = pylab.arange(min(combinedInten), max(combinedInten), 0.001)
#     smoothResp = fit.eval(smoothInt)
#     thresh = fit.inverse(0.8)
#     print(thresh)

#     #plot curve
#     pylab.subplot(122)
#     pylab.plot(smoothInt, smoothResp, '-')
#     pylab.plot([thresh, thresh],[0,0.8],'--'); pylab.plot([0, thresh],\
#     [0.8,0.8],'--')
#     pylab.title('threshold = %0.3f' %(thresh))
#     #plot points
#     pylab.plot(combinedInten, combinedResp, 'o')
#     pylab.show()
    
    


# %%
