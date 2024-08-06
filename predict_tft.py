import argparse
import pickle
from typing import Dict, List, Tuple
from functools import partial
import copy
import numpy as np
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from tqdm import tqdm
import torch
from torch import optim
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, Subset
from tft_torch.tft import TemporalFusionTransformer
import tft_torch.loss as tft_loss
import json
import os 

class DictDataSet(Dataset):
    def __init__(self, array_dict: Dict[str, np.ndarray]):
        self.keys_list = []
        for k, v in array_dict.items():
            self.keys_list.append(k)
            if np.issubdtype(v.dtype, np.dtype('bool')):
                setattr(self, k, torch.ByteTensor(v))
            elif np.issubdtype(v.dtype, np.int8):
                setattr(self, k, torch.CharTensor(v))
            elif np.issubdtype(v.dtype, np.int16):
                setattr(self, k, torch.ShortTensor(v))
            elif np.issubdtype(v.dtype, np.int32):
                setattr(self, k, torch.IntTensor(v))
            elif np.issubdtype(v.dtype, np.int64):
                setattr(self, k, torch.LongTensor(v))
            elif np.issubdtype(v.dtype, np.float32):
                setattr(self, k, torch.FloatTensor(v))
            elif np.issubdtype(v.dtype, np.float64):
                setattr(self, k, torch.DoubleTensor(v))
            else:
                setattr(self, k, torch.FloatTensor(v))

    def __getitem__(self, index):
        return {k: getattr(self, k)[index] for k in self.keys_list}

    def __len__(self):
        return getattr(self, self.keys_list[0]).shape[0]
    
def recycle(iterable):
    while True:
        for x in iterable:
            yield x

def get_set_and_loaders(data_dict: Dict[str, np.ndarray],
                        shuffled_loader_config: Dict,
                        serial_loader_config: Dict,
                        ignore_keys: List[str] = None,
                        ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dataset = DictDataSet({k:v for k,v in data_dict.items() if (ignore_keys and k not in ignore_keys)})
    loader = torch.utils.data.DataLoader(dataset,**shuffled_loader_config)
    serial_loader = torch.utils.data.DataLoader(dataset,**serial_loader_config)

    return dataset,iter(recycle(loader)),serial_loader

class QueueAggregator(object):
    def __init__(self, max_size):
        self._queued_list = []
        self.max_size = max_size

    def append(self, elem):
        self._queued_list.append(elem)
        if len(self._queued_list) > self.max_size:
            self._queued_list.pop(0)

    def get(self):
        return self._queued_list
    
def process_batch(batch: Dict[str,torch.tensor],
                  model: nn.Module,
                  quantiles_tensor: torch.tensor,
                  device:torch.device):
    if is_cuda:
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device)

    batch_outputs = model(batch)
    labels = batch['target']

    predicted_quantiles = batch_outputs['predicted_quantiles']
    q_loss, q_risk, _ = tft_loss.get_quantiles_loss_and_q_risk(outputs=predicted_quantiles,
                                                              targets=labels,
                                                              desired_quantiles=quantiles_tensor)
    return q_loss, q_risk

def predict(data_path, weights, file_path, cardinalities_map):
    with open(data_path,'rb') as fp:
        data = pickle.load(fp)

    configuration = {'optimization':
                    {
                        'batch_size': {'training': 128, 'inference': 128},# both weere 64 before
                        'learning_rate': 0.001,#was 0.001
                        'max_grad_norm': 1.0,
                    }
                    ,
                    'model':
                    {
                        'dropout': 0.05,#was 0.05 before
                        'state_size': 130,
                        'output_quantiles': [0.1, 0.5, 0.9],
                        'lstm_layers': 4,#was 2
                        'attention_heads': 4 #was 4 #then 6
                    },
                    # these arguments are related to possible extensions of the model class
                    'task_type':'regression',
                    'target_window_start': None
                    }

    feature_map = data['feature_map']
    #cardinalities_map = data['categorical_cardinalities']
    cardinalities_map = cardinalities_map
    
    structure = {
        'num_historical_numeric': len(feature_map['historical_ts_numeric']),
        'num_historical_categorical': len(feature_map['historical_ts_categorical']),
        'num_static_numeric': len(feature_map['static_feats_numeric']),
        'num_static_categorical': len(feature_map['static_feats_categorical']),
        'num_future_numeric': len(feature_map['future_ts_numeric']),
        'num_future_categorical': len(feature_map['future_ts_categorical']),
        'historical_categorical_cardinalities': [cardinalities_map[feat] + 1 for feat in feature_map['historical_ts_categorical']],
        'static_categorical_cardinalities': [cardinalities_map[feat] + 1 for feat in feature_map['static_feats_categorical']],
        'future_categorical_cardinalities': [cardinalities_map[feat] + 1 for feat in feature_map['future_ts_categorical']],
    }
    
    configuration['data_props'] = structure
    model = TemporalFusionTransformer(config=OmegaConf.create(configuration))
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    model.to(device)
    opt = optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                    lr=configuration['optimization']['learning_rate'])


    shuffled_loader_config = {'batch_size': configuration['optimization']['batch_size']['training'],
                    'drop_last': True,
                    'shuffle':False}

    serial_loader_config = {'batch_size': configuration['optimization']['batch_size']['inference'],
                    'drop_last': False,
                    'shuffle':False}

    # the following fields do not contain actual data, but are only identifiers of each observation
    meta_keys = ['time_index','combination_id', 'id', "Unnamed: 0", 'time', 'lat', 'lon', 'location']

    test_set,test_loader,test_serial_loader = get_set_and_loaders(data['data_sets']['test'],
                                                                    serial_loader_config,
                                                                    serial_loader_config,
                                                                    ignore_keys=meta_keys)

    shuffled_loader_config = {'batch_size': 128,
                    'drop_last': True,
                    'shuffle':False}

    serial_loader_config = {'batch_size': 128,
                    'drop_last': False,
                    'shuffle':False}

    # the following fields do not contain actual data, but are only identifiers of each observation
    test_set,test_loader,test_serial_loader = get_set_and_loaders(data['data_sets']['test'],
                                                                    serial_loader_config,
                                                                    serial_loader_config,
                                                                    ignore_keys=meta_keys)

    model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
    model.eval() # switch to evaluation mode

    output_aggregator = dict() # will be used for aggregating the outputs across batches

    with torch.no_grad():
        # go over the batches of the serial data loader
        for batch in tqdm(test_serial_loader):# change this from validation serial loader
            # process each batch
            if is_cuda:
                for k in list(batch.keys()):
                    batch[k] = batch[k].to(device)
            batch_outputs = model(batch)

            # accumulate outputs, as well as labels
            for output_key,output_tensor in batch_outputs.items():
                output_aggregator.setdefault(output_key,[]).append(output_tensor.cpu().numpy())
            
    validation_outputs = dict()
    for k in list(output_aggregator.keys()):
        validation_outputs[k] = np.concatenate(output_aggregator[k],axis=0)

    # Save the dictionary using Pickle
    with open(file_path, "wb") as pickle_file:
        pickle.dump(validation_outputs, pickle_file)


if __name__ == "__main__":
    
    data_d='../../../glab/users/al4385/data/TFT_30/'
    data_directory = '../../../glab/users/al4385/data/TFT_30_test/'
    weight_directory =  '../../../glab/users/al4385/weights/TFT_30_0429/'
    prediction_path = '../../../glab/users/al4385/predictions/TFT_30_0516/'
    filenames= ['BDT_-20_20',  'BDT_-20_-60', 'BDT_50_20',  'BDT_50_90',  'BET',  'NDT',  'NET']



    for i in filenames:
        data_or = data_d+i+'.pickle'
        with open(data_or,'rb') as fp:
            data_o = pickle.load(fp)
        cardinalities_map =  data_o['categorical_cardinalities']
        
        data_path = data_directory+i+'.pickle'
        weights = weight_directory+'weights_'+i+'.pth'
        file_path = prediction_path+'pred_'+i+'.pkl'
        predict(data_path, weights, file_path, cardinalities_map)