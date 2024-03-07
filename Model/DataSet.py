import pandas as pd

from Model.DataSetBinaryQuestionnaire import *
class DataSet():
    a = 0


def extract_data(csv_file_path):
    df = pd.read_csv(csv_file_path).values 
    return df    

def type_of_data(data, choice, cuts=None, points=None):
    if choice == 1:
        # use get_questionnaires to get the data
        dataHolder = cut_generator_binary(data)
        data = remove_cost_and_id(dataHolder)
        return order_cuts_by_cost(data)
    # elif choice == 2: