import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
import pickle

SAVE_MODEL_PATH = "src/models" 

def extract_tokens(data):
    list_all_token_dicts = []

    for nota_id in range(len(data)):
        list_all_token_dicts.extend(data[nota_id]['token_boxes'])
        
    return pd.DataFrame(list_all_token_dicts)

def pre_process(data: dict, train_flow: bool):
    
    tokens_df = extract_tokens(data) if train_flow else pd.DataFrame(data=data)

    tokens_df.drop(columns=['top', 'left', 'box', 'box_polygon', 'box_area', 
                    'box_height', 'box_width', 'x_position', 'y_position', 
                    'x_position_normalized', 'y_position_normalized',
                    'text', 'id_line_group'],
                    inplace=True)
    
    return (tokens_df.drop(columns=['label']), tokens_df['label']) if train_flow else tokens_df

def train_tree_model(data_train, data_test):
    X_train, y_train = pre_process(data_train, train_flow=True)

    tree_pipeline = make_pipeline(MinMaxScaler(), ExtraTreesClassifier(criterion='gini', n_estimators=10000, bootstrap=False))
    tree_pipeline.fit(X_train, y_train)
    
    X_test, y_test = pre_process(data_test, train_flow=True)
    y_hat = tree_pipeline.predict(X_test)
    report = classification_report(y_test, y_hat)
    print('Resultado de entrenamiento de Decision Tree Classifier:')
    print(report)

    return tree_pipeline

def load_tree(model_path=SAVE_MODEL_PATH):
    with open(model_path + '/tree_model.pkl', 'rb') as f:
        tree_pipeline = pickle.load(f)
    
    return tree_pipeline