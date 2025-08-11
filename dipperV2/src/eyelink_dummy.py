class DummyEyeTracker:
    def __init__(self, *args, **kwargs):
        print("Dummy eye tracker initialized (tracking disabled)")

    def startTracker(self):
        print("Started tracking")
        
    def closeTracker(self):
        print("Closed tracker")

    def stimOnset(self,trial_id,condition,contrast):
        print("Sent message for stimOnset")
        
    def logResponse(self,response,rt):
        print('Response logged')

