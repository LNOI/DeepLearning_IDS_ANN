from typing import NamedTuple
from kfp.components import InputPath, OutputPath
from kfp.components import func_to_container_op

from datetime import datetime

import sys
sys.path.insert(0,"..")

def prepare_data(
    url : str,
    X_train_path:  OutputPath("PKL"),
    Y_train_path:  OutputPath("PKL"),
    X_test_path:  OutputPath("PKL"),
    Y_test_path:  OutputPath("PKL")
):  
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
    import joblib
    import wget

    wget.download(url)
    cols="""duration,
    protocol_type,
    service,
    flag,
    src_bytes,
    dst_bytes,
    land,
    wrong_fragment,
    urgent,
    hot,
    num_failed_logins,
    logged_in,
    num_compromised,
    root_shell,
    su_attempted,
    num_root,
    num_file_creations,
    num_shells,
    num_access_files,
    num_outbound_cmds,
    is_host_login,
    is_guest_login,
    count,
    srv_count,
    serror_rate,
    srv_serror_rate,
    rerror_rate,
    srv_rerror_rate,
    same_srv_rate,
    diff_srv_rate,
    srv_diff_host_rate,
    dst_host_count,
    dst_host_srv_count,
    dst_host_same_srv_rate,
    dst_host_diff_srv_rate,
    dst_host_same_src_port_rate,
    dst_host_srv_diff_host_rate,
    dst_host_serror_rate,
    dst_host_srv_serror_rate,
    dst_host_rerror_rate,
    dst_host_srv_rerror_rate"""

    columns=[]
    for c in cols.split(','):
        if(c.strip()):
            columns.append(c.strip())
    columns.append('target')

    attacks_types = {
    'normal': 'normal',
    'back': 'dos',
    'buffer_overflow': 'u2r',
    'ftp_write': 'r2l',
    'guess_passwd': 'r2l',
    'imap': 'r2l',
    'ipsweep': 'probe',
    'land': 'dos',
    'loadmodule': 'u2r',
    'multihop': 'r2l',
    'neptune': 'dos',
    'nmap': 'probe',
    'perl': 'u2r',
    'phf': 'r2l',
    'pod': 'dos',
    'portsweep': 'probe',
    'rootkit': 'u2r',
    'satan': 'probe',
    'smurf': 'dos',
    'spy': 'r2l',
    'teardrop': 'dos',
    'warezclient': 'r2l',
    'warezmaster': 'r2l',
    }
    path = "kddcup.data_10_percent.gz"
    df = pd.read_csv(path,names=columns)

    #Adding Attack Type column
    df['Attack Type'] = df.target.apply(lambda r:attacks_types[r[:-1]])
    #Finding categorical features
    num_cols = df._get_numeric_data().columns

    cate_cols = list(set(df.columns)-set(num_cols))
    cate_cols.remove('target')
    cate_cols.remove('Attack Type')
    df = df.dropna('columns')# drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]]# keep columns where there are more than 1 unique values

    corr = df.corr()
    df.drop('num_root',axis = 1,inplace = True)
    df.drop('srv_serror_rate',axis = 1,inplace = True)
    df.drop('srv_rerror_rate',axis = 1, inplace=True)
    df.drop('dst_host_srv_serror_rate',axis = 1, inplace=True)
    df.drop('dst_host_serror_rate',axis = 1, inplace=True)
    df.drop('dst_host_rerror_rate',axis = 1, inplace=True)
    df.drop('dst_host_srv_rerror_rate',axis = 1, inplace=True)
    df.drop('dst_host_same_srv_rate',axis = 1, inplace=True)
    df_std = df.std()
    df_std = df_std.sort_values(ascending = True)
    #protocol_type feature mapping
    pmap = {'icmp':0,'tcp':1,'udp':2}
    df['protocol_type'] = df['protocol_type'].map(pmap)
    #flag feature mapping
    fmap = {'SF':0,'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5 ,'S1':6 ,'S2':7,'RSTOS0':8,'S3':9 ,'OTH':10}
    df['flag'] = df['flag'].map(fmap)
    df.drop('service',axis = 1,inplace= True)
    df = df.drop(['target',], axis=1)
    print(df.shape)

    # Target variable and train set
    Y = df[['Attack Type']]
    X = df.drop(['Attack Type',], axis=1)

    sc = MinMaxScaler()
    X = sc.fit_transform(X)

    # Split test and train data 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    joblib.dump(X_train,X_train_path)
    joblib.dump(Y_train,Y_train_path)
    joblib.dump(X_test,X_test_path)
    joblib.dump(Y_test,Y_test_path)
    
    print(X_train.shape, X_test.shape)
    print(Y_train.shape, Y_test.shape)
    
prepare_data_op = func_to_container_op(
    func =  prepare_data ,
    packages_to_install = [
        "joblib",
        "keras",
        "tensorflow",
        "wget",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "numpy",
        "seaborn"
    ]   
)


def train_data(
    X_train_path:  InputPath("PKL"),
    Y_train_path:  InputPath("PKL"),
    X_test_path:  InputPath("PKL"),
    Y_test_path:  InputPath("PKL"),
    model_dir : OutputPath(str)
):  
    import os
    import time
    import tensorflow as tf
    from tensorflow import keras
    import joblib
    import wget
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    lb = LabelEncoder()
    X_train = joblib.load(X_train_path)
    Y_train = joblib.load(Y_train_path)
    X_test = joblib.load(X_test_path)
    Y_test = joblib.load(Y_test_path)
        
    Y_train = lb.fit_transform(Y_train)
    Y_test = lb.fit_transform(Y_test)
    def create_model():
      model = tf.keras.models.Sequential([
          keras.layers.Dense(30, activation='relu', input_dim=30,kernel_initializer='random_uniform'),
          keras.layers.Dense(1,activation='sigmoid',kernel_initializer='random_uniform'),
          keras.layers.Dense(5, activation='softmax')
      ])
      model.compile(optimizer='adam',
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=[tf.metrics.SparseCategoricalAccuracy()])
      return model

    model = create_model()
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    
    model.fit(X_train, 
          Y_train,  
          epochs=1,
          batch_size=64,
          validation_data=(X_test, Y_test))
    
    model.save(model_dir)
    print(f"Model saved {model_dir}")
    print(os.listdir(model_dir))
    
train_op = func_to_container_op(
    func = train_data  ,
    packages_to_install = [
        "joblib",
        "tensorflow",
        "wget",
        "scikit-learn",
        "pandas",
    ]   
)

def evaluate_model(
    model_dir : InputPath(str),
    X_test_path:  InputPath("PKL"),
    Y_test_path:  InputPath("PKL")
) -> NamedTuple("EvaluationOutput", [("mlpipeline_metrics", "Metrics")]):
    import os
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import LabelEncoder
    import joblib
    from collections import namedtuple
    lb = LabelEncoder()
    X_test = joblib.load(X_test_path)
    Y_test = joblib.load(Y_test_path)
    Y_test = lb.fit_transform(Y_test)
    model = tf.keras.models.load_model(model_dir)
    model.summary()
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)
    metrics = {
        "metrics": [
            {"name": "loss", "numberValue": str(loss), "format": "PERCENTAGE"},
            {"name": "accuracy", "numberValue": str(accuracy), "format": "PERCENTAGE"},
        ]
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    out_tuple = namedtuple("EvaluationOutput", ["mlpipeline_metrics"])
    return out_tuple(json.dumps(metrics))   
evl_op = func_to_container_op(
    func = evaluate_model,
    packages_to_install = [
        "joblib",
        "tensorflow",
        "scikit-learn"
    ]   
)

def monitoring(
):
    pass
monitoring_op = func_to_container_op(
    func = monitoring  ,
    packages_to_install = [
        "joblib",
        "keras",
        "wget",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "numpy",
        "seaborn"
    ]   
)

import kfp
from kfp import dsl
@dsl.pipeline(
  name='Deep Learning IDS/IPS',
  description='Pipeline'
)
def my_pipeline(url):
    prepare_data_task = prepare_data_op(url)
    train_task = train_op(x_train= prepare_data_task.outputs["X_train"],
                          y_train= prepare_data_task.outputs["Y_train"],
                          x_test= prepare_data_task.outputs["X_test"],
                          y_test= prepare_data_task.outputs["Y_test"],
                          ) 
    evl_task = evl_op(model_dir=train_task.outputs["model_dir"],
                          x_test= prepare_data_task.outputs["X_test"],
                          y_test= prepare_data_task.outputs["Y_test"]
                     ) 
    # monitoring_task = monitoring_op() 

session_cookies = "MTY3MDM3NzgxNHxOd3dBTkVsWFUwOVpORUZMV1VkWVZWVlhRazQyTjBGSVdrVklURXREVFVnMU4wWlNXRTVMVVVwUE4xY3lTRmxUTWxWSU5WUmFTVkU9fNsIv8kMYn9FxRmsuD6HJPn8tqXZDuYXmIkjQuj4qn0r"
HOST = "http://10.64.140.43.nip.io"
client = kfp.Client(
  host= f"{HOST}/pipeline",
  cookies = f"authservice_session={session_cookies}",
  namespace="admin"
)
redirect = 0
if redirect:
    client.create_run_from_pipeline_func(
      my_pipeline,
      arguments={
        'url': 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
      }
    )
else:
    kfp.compiler.Compiler().compile(
        pipeline_func = my_pipeline,
        package_path = "pipeline_dnn.yaml"
    )