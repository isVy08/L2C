from data_generator import DataGenerator
from blackbox import load_linear_classifier, load_neural_classifier, LinearClassifier

def load_data(name, discretized, device, strategy):
  print('Loading:',name,"- Discretized:", discretized)
  if name == 'german':
    num_cols = ['Months', 'Credit-amount', 'age']
    target_col = 'target'
    immutable_cols = ['Foreign-worker', 'Number-of-people-being-lible', 'Personal-status', 'Purpose']
    causal_relations = ['age', 'Present-employment-since', 'Present-residence-since', 'Months']
    quasi_identifiers = ['age', 'Foreign-worker', 'Personal-status', 
                        'Present-residence-since', 
                        'Present-employment-since',
                        'Job', 'Property', 'Housing' 
                        ]

  
  elif name == 'admission':
    num_cols = ['GRE Score', 'TOEFL Score', 'CGPA']
    target_col = 'Chance of Admit'
    immutable_cols = ['University Rating']
    causal_relations = ['Research']
    quasi_identifiers = None

  elif name == 'student':
    num_cols = ['age', 'absences', 'G1', 'G2']
    target_col = 'label'
    immutable_cols = ['Medu', 'Fedu', 'famsup', 'G1']
    causal_relations = ['age']
    quasi_identifiers = None

  elif name == 'sba':
    target_col = 'label'
    num_cols = ['Term', 'NoEmp', 'CreateJob', 'RetainedJob', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv', 'Portion']
    immutable_cols = ['UrbanRural', 'New', 'Recession']
    causal_relations = ['Term']
    quasi_identifiers = None
  
  elif name == 'adult':
    target_col = 'target'
    num_cols = ['age', 'capitalgain', 'capitalloss', 'hoursperweek']
    immutable_cols = ['race', 'sex', 'native-country', 'marital-status']
    causal_relations = ['age', 'education']
    quasi_identifiers = ['age', 'sex', 'race', 'relationship', 'marital-status']
  
  else:
    raise ValueError

  
  
  dg = DataGenerator(name, num_cols, target_col, immutable_cols, 
                discretized, device, strategy, causal_relations, 
                quasi_identifiers)
  return dg, immutable_cols

def load_blackbox(name, dg, wrap_linear=False):
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
