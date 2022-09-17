from data_generator import DataGenerator
from blackbox import load_linear_classifier, load_neural_classifier, LinearClassifier

def load_data(name, discretized, device, nbins=None):
  print('Loading:',name,"- Discretized:", discretized)
  if name == 'german':
    num_dict = {'Months': 3, 'Credit-amount': 3, 'age': 3}
    target_col = 'target'
    immutable_cols = ['Foreign-worker', 'Number-of-people-being-lible', 'Personal-status', 'Purpose']
    
  
  elif name == 'admission':
    num_dict = {'GRE Score': 3, 'TOEFL Score': 3, 'CGPA': 3}
    target_col = 'Chance of Admit'
    immutable_cols = ['University Rating']

  elif name == 'student':
    num_dict = {'age': 2, 'absences': 3, 'G1': 3, 'G2': 4}
    target_col = 'label'
    immutable_cols = ['Medu', 'Fedu', 'famsup', 'G1']

  elif name == 'sba':
    target_col = 'label'
    num_dict = {'Term': 4, 'NoEmp': 4, 'CreateJob': 4, 'RetainedJob': 3, 'ChgOffPrinGr': 4, 'GrAppv': 4, 'SBA_Appv': 4, 'Portion': 3}
    immutable_cols = ['UrbanRural', 'New', 'Recession']
  
  else:
    raise ValueError('This dataset is not supported!')

  if nbins is not None:
    for col in num_dict:
      num_dict[col] = nbins
  
  dg = DataGenerator(name, num_dict, target_col, immutable_cols, discretized, device)
  return dg, immutable_cols

def load_blackbox(name, dg, wrap_linear=False):
  '''
  Wrap sklearn Logistic Regression model into a differentiable MLP
  '''
  classifiers = []
  for cls_index in range(1,6):

    if name in ('german', 'student'):
      cls = load_linear_classifier(dg, cls_index)
      if wrap_linear:
        cls = LinearClassifier(cls, dg.device)
    else:
      cls = load_neural_classifier(dg, cls_index, 100, dg.device)
    
    classifiers.append(cls)
  
  return classifiers
