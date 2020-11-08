#19-04-Absenteeism(Integration)

from AbsenteeismModule import *
pd.read_csv('AbsenteeismNewData.csv')
model = absenteeism_model('model', 'scaler')
model.load_and_clean_data('AbsenteeismNewData.csv')
model.predicted_outputs()
model.predicted_outputs().to_csv('AbsenteeismPredictions.csv', index = False)