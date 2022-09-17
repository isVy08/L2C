import logging
import numpy as np
import pandas as pd
import shutil
import os
import dice_ml

from sklearn.model_selection import train_test_split

from utils.validation import check_random_state


def load_adult_income_dataset(filepath=None, only_train=False):
    """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares
       the data for data analysis based on https://rpubs.com/H_Zhu/235617
    :return adult_data: returns preprocessed adult income dataset.
    """
    #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']

    if filepath is None:
        raw_data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                                 delimiter=', ', dtype=str, invalid_raise=False)

        adult_data = pd.DataFrame(raw_data, columns=column_names)
    else:
        adult_data = pd.read_csv(filepath, names=column_names, dtype=str)

        df_obj = adult_data.select_dtypes(['object'])
        adult_data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data = adult_data.astype(
        {"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64})

    adult_data = adult_data.replace(
        {'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government',
                                     'Local-gov': 'Government'}})
    adult_data = adult_data.replace(
        {'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    adult_data = adult_data.replace(
        {'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

    adult_data = adult_data.replace(
        {
            'occupation': {
                'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                'Exec-managerial': 'White-Collar', 'Farming-fishing': 'Blue-Collar',
                'Handlers-cleaners': 'Blue-Collar',
                'Machine-op-inspct': 'Blue-Collar', 'Other-service': 'Service',
                'Priv-house-serv': 'Service',
                'Prof-specialty': 'Professional', 'Protective-serv': 'Service',
                'Tech-support': 'Service',
                'Transport-moving': 'Blue-Collar', 'Unknown': 'Other/Unknown',
                'Armed-Forces': 'Other/Unknown', '?': 'Other/Unknown'
            }
        }
    )

    adult_data = adult_data.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
                                                        'Married-spouse-absent': 'Married', 'Never-married': 'Single'}})

    adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                              'Amer-Indian-Eskimo': 'Other'}})

    adult_data = adult_data[['age', 'workclass', 'education', 'marital-status', 'occupation',
                             'race', 'gender', 'hours-per-week', 'income']]

    adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

    adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                                   '11th': 'School', '10th': 'School', '7th-8th': 'School',
                                                   '9th': 'School', '12th': 'School', '5th-6th': 'School',
                                                   '1st-4th': 'School', 'Preschool': 'School'}})

    adult_data = adult_data.rename(
        columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

    if only_train:
        train, _ = train_test_split(adult_data, test_size=0.2, random_state=17)
        adult_data = train.reset_index(drop=True)

    # Remove the downloaded dataset
    if os.path.isdir('archive.ics.uci.edu'):
        entire_path = os.path.abspath('archive.ics.uci.edu')
        shutil.rmtree(entire_path)

    adult_data = adult_data.rename(columns={"income": "label"})
    continuous_features = ['age', 'hours_per_week']

    return adult_data, continuous_features


def gen_synthetic_data(num_samples=1000, mean_0=None, cov_0=None, mean_1=None, cov_1=None, random_state=None):
    random_state = check_random_state(random_state)

    if mean_0 is None or cov_0 is None or mean_1 is None or cov_1 is None:
        mean_0 = np.array([-2, -2])
        cov_0 = np.array([[0.5, 0], [0, 0.5]])
        mean_1 = np.array([2, 2])
        cov_1 = np.array([[0.5, 0], [0, 0.5]])

    num_class0 = random_state.binomial(n=num_samples, p=0.5)
    x_class0 = random_state.multivariate_normal(mean_0, cov_0, num_class0)
    x_class1 = random_state.multivariate_normal(
        mean_1, cov_1, num_samples-num_class0)
    data0 = np.hstack([x_class0, np.zeros((num_class0, 1))])
    data1 = np.hstack([x_class1, np.ones((num_samples-num_class0, 1))])
    raw_data = np.vstack([data0, data1])
    random_state.shuffle(raw_data)
    column_names = ['f' + str(i) for i in range(len(mean_0))] + ['label']
    df = pd.DataFrame(raw_data, columns=column_names)
    return df


def get_dataset(dataset='synthesis', params=None):
    if 'synthesis' in dataset:
        dataset = gen_synthetic_data(*params)
        numerical = list(dataset.columns)
        numerical.remove('label')
    elif 'german' in dataset:
        dataset = pd.read_csv('./data/corrected_german_small.csv'
                              if 'shift' in dataset else './data/german_small.csv')
        # numerical = ['Duration', 'Credit amount', 'Installment rate',
                     # 'Present residence', 'Age', 'Existing credits', 'Number people']
        numerical = ['Duration', 'Credit amount', 'Age']
    elif 'sba' in dataset:
        dataset = pd.read_csv('./data/sba_shift.csv'
                              if 'shift' in dataset else './data/sba.csv')
        categorical = ['LowDoc', 'RevLineCr', 'NewExist',
                       'MIS_Status', 'UrbanRural', 'label']
        numerical = list(dataset.columns.difference(categorical))
    elif 'student' in dataset:
        dataset = pd.read_csv('./data/ms_student.csv'
                              if 'shift' in dataset else './data/gp_student.csv')
        numerical = ['Fedu', 'G1', 'G2', 'Medu', 'absences',
                     'age', 'freetime', 'goout', 'health', 'studytime']
    else:
        raise ValueError("Unknown dataset")

    return dataset, numerical


def get_full_dataset(dataset='synthesis', params=list()):
    if 'synthesis' in dataset:
        joint_dataset = gen_synthetic_data(*params)
        numerical = list(joint_dataset.columns)
        numerical.remove('label')
    elif 'german' in dataset:
        dataset = pd.read_csv('./data/german_small.csv')
        shift_dataset = pd.read_csv('./data/corrected_german_small.csv')
        joint_dataset = dataset.append(shift_dataset)
        # numerical = ['Duration', 'Credit amount', 'Installment rate',
                     # 'Present residence', 'Age', 'Existing credits', 'Number people']
        numerical = ['Duration', 'Credit amount', 'Age']
    elif 'sba' in dataset:
        dataset = pd.read_csv('./data/sba.csv')
        shift_dataset = pd.read_csv('./data/sba_shift.csv')
        joint_dataset = dataset.append(shift_dataset)
        categorical = ['LowDoc', 'RevLineCr', 'NewExist',
                       'MIS_Status', 'UrbanRural', 'label']
        numerical = list(dataset.columns.difference(categorical))
    elif 'student' in dataset:
        dataset = pd.read_csv('./data/gp_student.csv')
        shift_dataset = pd.read_csv('./data/ms_student.csv')
        joint_dataset = dataset.append(shift_dataset)
        numerical = ['Fedu', 'G1', 'G2', 'Medu', 'absences',
                     'age', 'freetime', 'goout', 'health', 'studytime']
    else:
        raise ValueError("Unknown dataset")

    return joint_dataset, numerical


def make_logger(name, log_dir):
    log_dir = log_dir or 'logs'
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'debug.log')
    handler = logging.FileHandler(log_file)
    formater = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formater)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formater)
    logger.addHandler(stream_handler)

    return logger

