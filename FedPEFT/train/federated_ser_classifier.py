from moviepy.tools import verbose_print
import torch
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
import torchaudio
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from collections import OrderedDict

import numpy as np
from pathlib import Path
import pandas as pd
import copy, time, pickle, shutil, sys, os, pdb
from copy import deepcopy
import loralib as lora
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import OrderedDict

##【！！！重要】程序无法访问hugggingface.co时使用替代方案
##chenhj
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model'))

from dnn_models import dnn_classifier
from wav2vec import Wav2VecWrapper, ShallowModel, DeepModel
from wavlm_plus import WavLMWrapper
from whisper01 import WhisperWrapper
from update import average_weights, average_gradients, local_trainer
from audiomentations import Compose, AddBackgroundNoise, PolarityInversion, AddGaussianSNR, TimeMask, TimeStretch


# define label mapping
#emo_dict = {'neu': 0, 'hap': 1, 'sad': 2, 'ang': 3}
emo_dict = {'neu': 0, 'hap': 3, 'sad': 1, 'ang': 2}

affect_dict = {'low': 0, 'med': 1, 'high': 2}
gender_dict = {'F': 0, 'M': 1}
                       
# define feature len mapping
feature_len_dict = {'emobase': 988, 'ComParE': 6373, 'wav2vec': 9216, 
                    'apc': 512, 'distilhubert': 768, 'tera': 768, 'wav2vec2': 768,
                    'decoar2': 768, 'cpc': 256, 'audio_albert': 768, 
                    'mockingjay': 768, 'npc': 512, 'vq_apc': 512, 'vq_wav2vec': 512}

def save_result(save_index, acc, uar, best_epoch, dataset):
    row_df = pd.DataFrame(index=[save_index])
    row_df['acc'], row_df['uar'], row_df['epoch'], row_df['dataset']  = acc, uar, best_epoch, dataset
    return row_df


def collate_fn(batch):
    # max of 6s of data
    max_audio_len = min(max([b[0].shape[0] for b in batch]), 16000 * 6)
    data, len_data, taregt, source_ids = list(), list(), list(), list()
    for idx in range(len(batch)):
        # append data
        data.append(padding_cropping(batch[idx][0], max_audio_len))

        # append len
        if len((batch[idx][0])) >= max_audio_len:
            len_data.append(torch.tensor(max_audio_len))
        else:
            len_data.append(torch.tensor(len((batch[idx][0]))))

        # append target
        taregt.append(torch.tensor(batch[idx][1]))
        # append source_id
        source_ids.append(batch[idx][2])

    data = torch.stack(data, dim=0)
    len_data = torch.stack(len_data, dim=0)
    taregt = torch.stack(taregt, dim=0)

    return data, taregt, len_data, source_ids

def collate_fn01(batch):
    # max of 6s of data
    max_audio_len = min(max([b[0].shape[0] for b in batch]), 16000 * 6)
    data, len_data, taregt = list(), list(), list()
    for idx in range(len(batch)):
        # append data
        data.append(padding_cropping(batch[idx][0], max_audio_len))

        # append len
        if len((batch[idx][0])) >= max_audio_len:
            len_data.append(torch.tensor(max_audio_len))
        else:
            len_data.append(torch.tensor(len((batch[idx][0]))))

        # append target
        taregt.append(torch.tensor(batch[idx][1]))

    data = torch.stack(data, dim=0)
    len_data = torch.stack(len_data, dim=0)
    taregt = torch.stack(taregt, dim=0)
    return data, taregt, len_data

def padding_cropping(
    input_wav, size
):
    if len(input_wav) > size:
        input_wav = input_wav[:size]
    elif len(input_wav) < size:
        input_wav = torch.nn.ConstantPad1d(padding=(0, size - len(input_wav)), value=0)(input_wav)
    return input_wav

#chenhj 该函数会遍历模型中的所有参数，并仅返回那些 requires_grad=True 的参数
def get_trainable_params(model):
    """
    Returns a state dict containing only the parameters that require gradients.
    """
    trainable_state_dict = OrderedDict((name, param.clone().detach()) for name, param in model.named_parameters() if param.requires_grad)
    # trainable_state_dict = OrderedDict((k, v) for k, v in model.state_dict().items() if v.requires_grad)
    # trainable_state_dict = {name: param for name, param in model.state_dict().items() if model.get_parameter(name).requires_grad}
    return trainable_state_dict

#训练集处理方法
class DatasetGenerator():
    def __init__(self, dataset, audio_duration: int=6, is_train: bool=False):
        self.dataset = dataset
        self.audio_duration = audio_duration
        self.is_train = is_train

        self.transform = Compose([
            AddGaussianSNR(min_snr_in_db=10.0, max_snr_in_db=30.0, p=1.0),
            TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1.0)
        ])

    def __len__(self):
        return len(self.dataset['dataset'])

    def __getitem__(self, item):
        data, _ = torchaudio.load(self.dataset['data'][item])
        data = data[0]
        if data.isnan()[0].item(): data = torch.zeros(data.shape)
        if len(data) > self.audio_duration * 16000: data = data[:self.audio_duration * 16000]
        if self.is_train:
            data = data.detach().cpu().numpy()
            data = self.transform(samples=data, sample_rate=16000)
            data = torch.tensor(data)
        return data, int(self.dataset['label'][item]), self.dataset['source_id'][item]

#验证集和测试集处理方法
class DatasetGenerator01():
    def __init__(self, dataset, audio_duration: int=6, is_train: bool=False):
        self.dataset = dataset
        self.audio_duration = audio_duration
        self.is_train = is_train

        self.transform = Compose([
            AddGaussianSNR(min_snr_in_db=10.0, max_snr_in_db=30.0, p=1.0),
            TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1.0)
        ])

    def __len__(self):
        return len(self.dataset['dataset'])

    def __getitem__(self, item):
        data, _ = torchaudio.load(self.dataset['data'][item])
        data = data[0]
        if data.isnan()[0].item(): data = torch.zeros(data.shape)
        if len(data) > self.audio_duration * 16000: data = data[:self.audio_duration * 16000]
        if self.is_train:
            data = data.detach().cpu().numpy()
            data = self.transform(samples=data, sample_rate=16000)
            data = torch.tensor(data)
        return data, int(self.dataset['label'][item])

def read_data_dict_by_client(dataset_list, fold_idx):
    
    return_train_dict, return_val_dict, return_test_dict = {}, {}, {}
    dataset_label_list = []

    # prepare the data for the training
    for dataset in dataset_list:
        with open(preprocess_path.joinpath(dataset, 'fold'+str(int(fold_idx+1)), 'training_'+args.norm+'.pkl'), 'rb') as f:
            train_dict = pickle.load(f)
        with open(preprocess_path.joinpath(dataset, 'fold'+str(int(fold_idx+1)), 'test_'+args.norm+'.pkl'), 'rb') as f:
            test_dict = pickle.load(f)
        for tmp_dict in [train_dict, test_dict]:
            for key in tmp_dict: tmp_dict[key]['dataset'] = dataset

        # test set will be the same
        x_test = []
        y_test = np.zeros([len(test_dict)])
        for key_idx, key in enumerate(list(test_dict.keys())):
            x_test.append(str(test_dict[key]['file_path']))
            y_test[key_idx] = int(emo_dict[test_dict[key]['label']])
            dataset_label_list.append(test_dict[key]['dataset'])
        
        if len(return_test_dict) == 0:
            return_test_dict['data'], return_test_dict['label'] = x_test, y_test
            return_test_dict['dataset'] = dataset_label_list
        else:
            return_test_dict['data'] = np.append(return_test_dict['data'], x_test, axis=0)
            return_test_dict['label'] = np.append(return_test_dict['label'], y_test, axis=0)
            return_test_dict['dataset'] = dataset_label_list

        # we remake the data dict per speaker for the ease of local training
        train_speaker_data_dict = {}
        for key in train_dict:
            speaker_id = str(train_dict[key]['speaker_id'])
            if speaker_id not in train_speaker_data_dict: train_speaker_data_dict[speaker_id] = []
            train_speaker_data_dict[speaker_id].append(key)
            
        # in federated setting that the norm validation dict will be based on local client data
        # so we combine train_dict and validate_dict in centralized setting
        # then choose certain amount data per client as local validation set
        if dataset == 'crema-d':
            for speaker_id in train_speaker_data_dict:
                speaker_data_key_list = train_speaker_data_dict[speaker_id]
                x, y = np.zeros([len(speaker_data_key_list), feature_len_dict[args.feature_type]]), np.zeros([len(speaker_data_key_list)])
                dataset_list = []
                x = []
                for idx, data_key in enumerate(speaker_data_key_list):
                    x.append(str(train_dict[data_key]['file_path']))
                    y[idx] = int(emo_dict[train_dict[data_key]['label']])
                    dataset_list.append(train_dict[data_key]['dataset'])
                np.random.seed(8)
                idx_array = np.random.permutation(len(train_speaker_data_dict[speaker_id]))
                perm_array = np.random.permutation(len(idx_array))
                return_train_dict[speaker_id] = {}
                return_train_dict[speaker_id]['data'] = [x[idx] for idx in perm_array[:int(0.8*len(x))]]
                return_train_dict[speaker_id]['label'] = y[perm_array[:int(0.8*len(x))]].copy()
                return_train_dict[speaker_id]['gender'] = train_dict[data_key]['gender']
                return_train_dict[speaker_id]['dataset'] = [dataset_list[idx] for idx in perm_array[:int(0.8*len(x))]]

                # We save the utterance keys used for training and validation per speaker (client)
                return_val_dict[speaker_id] = {}
                return_val_dict[speaker_id]['data'] = [x[idx] for idx in perm_array[int(0.8*len(x)):]]
                return_val_dict[speaker_id]['label'] = y[perm_array[int(0.8*len(x)):]].copy()
                return_val_dict[speaker_id]['gender'] = train_dict[data_key]['gender']
                return_val_dict[speaker_id]['dataset'] = [dataset_list[idx] for idx in perm_array[int(0.8*len(x)):]]

        else:
            # we want to divide speaker data if the dataset is iemocap or msp-improv to increase client size
            for speaker_id in train_speaker_data_dict:
                # in iemocap and msp-improv
                # we spilit each speaker data into 10 parts in order to create more clients
                np.random.seed(8)
                idx_array = np.random.permutation(len(train_speaker_data_dict[speaker_id]))
                speaker_data_key_list = train_speaker_data_dict[speaker_id]
                split_array = np.array_split(idx_array, 10)
                for split_idx in range(len(split_array)):
                    # we randomly pick 10% of data
                    idxs_train = split_array[split_idx]

                    y = np.zeros([len(idxs_train)])
                    dataset_list = []
                    x = []
                    for idx, key_idx in enumerate(idxs_train):
                        data_key = speaker_data_key_list[key_idx]
                        x.append(str(train_dict[data_key]['file_path']))
                        y[idx] = int(emo_dict[train_dict[data_key]['label']])
                        dataset_list.append(train_dict[data_key]['dataset'])
                        # if speaker_id+'_'+str(split_idx) not in return_train_dict: return_train_dict[speaker_id+'_'+str(split_idx)] = {}
                        # return_train_dict[speaker_id+'_'+str(split_idx)][key] = train_dict[key].copy()
                    np.random.seed(8)
                    perm_array = np.random.permutation(len(idxs_train))
                    return_train_dict[speaker_id+'_'+str(split_idx)] = {}
                    return_train_dict[speaker_id+'_'+str(split_idx)]['data'] = [x[idx] for idx in perm_array[:int(0.8*len(x))]]
                    return_train_dict[speaker_id+'_'+str(split_idx)]['label'] = y[perm_array[:int(0.8*len(x))]].copy()
                    return_train_dict[speaker_id+'_'+str(split_idx)]['gender'] = train_dict[data_key]['gender']
                    return_train_dict[speaker_id+'_'+str(split_idx)]['dataset'] = [dataset_list[idx] for idx in perm_array[:int(0.8*len(x))]]

                    # We save the utterance keys used for training and validation per speaker (client)
                    return_val_dict[speaker_id+'_'+str(split_idx)] = {}
                    return_val_dict[speaker_id+'_'+str(split_idx)]['data'] = [x[idx] for idx in perm_array[int(0.8*len(x)):]]
                    return_val_dict[speaker_id+'_'+str(split_idx)]['label'] = y[perm_array[int(0.8*len(x)):]].copy()
                    return_val_dict[speaker_id+'_'+str(split_idx)]['gender'] = train_dict[data_key]['gender']
                    return_val_dict[speaker_id+'_'+str(split_idx)]['dataset'] = [dataset_list[idx] for idx in perm_array[int(0.8*len(x)):]]

                    
    return return_train_dict, return_val_dict, return_test_dict

if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', default='iemocap')
    parser.add_argument('--feature_type', default='emobase')
    parser.add_argument('--dropout', default=0.2)
    parser.add_argument('--learning_rate', default=0.05, type=float)
    parser.add_argument('--batch_size', default=20)
    parser.add_argument('--use_gpu', default=True)
    parser.add_argument('--num_epochs', default=200)
    parser.add_argument('--local_epochs', default=1)
    parser.add_argument('--norm', default='znorm')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--model_type', default='fed_sgd')
    parser.add_argument('--pred', default='emotion')
    parser.add_argument('--save_dir', default='/media/data/projects/speech-privacy')

    #chenhj
    parser.add_argument('--pretrain_model', default='wav2vec2_0', type=str, help="pretrained model type")
    parser.add_argument('--finetune_method', default='lora', type=str, help="finetune method: adapter, embedding prompt, input prompt")
    parser.add_argument('--lora_rank', default=8, type=int, help='lora rank')
    parser.add_argument('--adapter_hidden_dim', default=128, type=int, help='adapter dimension')
    parser.add_argument('--embedding_prompt_dim', default=5, type=int, help='adapter dimension')
    parser.add_argument('--downstream_model', default='rnn', type=str, help="model type")
    parser.add_argument('--finetune_emb', default="all", type=str, help='adapter dimension')
    parser.add_argument('--use-conv-output', action='store_true', help='use conv output')
    parser.add_argument('--num_shallow_layers', default=6, type=int, help='Number of shallow layers for AFL updates')
    args = parser.parse_args()
    if args.finetune_method == "adapter" or args.finetune_method == "adapter_l":
        setting = f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.finetune_method}_{args.adapter_hidden_dim}'
    elif args.finetune_method == "embedding_prompt":
        setting = f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.finetune_method}_{args.embedding_prompt_dim}'
    elif args.finetune_method == "lora":
        setting = f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.finetune_method}_{args.lora_rank}'
    elif args.finetune_method == "finetune":
        setting = f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.finetune_method}'
    elif args.finetune_method == "combined":
        setting = f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.finetune_method}_{args.adapter_hidden_dim}_{args.embedding_prompt_dim}_{args.lora_rank}'
    args.setting = setting
    if args.finetune_emb != "all":
        args.setting = args.setting + "_avgtok"
    if args.use_conv_output:
        args.setting = args.setting + "_conv_output"

    preprocess_path = Path(args.save_dir).joinpath('federated_learning', args.feature_type, args.pred)
    
    # set seeds
    seed_everything(8, workers=True)
    save_result_df = pd.DataFrame()
    dataset_list = args.dataset.split('_')

    # find device
    device = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    # We perform 5 fold experiments
    for fold_idx in range(5):
        # save folder details
        save_row_str = 'fold'+str(int(fold_idx+1))
        row_df = pd.DataFrame(index=[save_row_str])
        
        model_setting_str = 'local_epoch_'+str(args.local_epochs) if args.model_type == 'fed_avg' else 'local_epoch_1'
        model_setting_str += '_dropout_' + str(args.dropout).replace('.', '')
        model_setting_str += '_lr_' + str(args.learning_rate)[2:]
        
        # Read the data per speaker
        train_speaker_dict, val_speaker_dict, test_speaker_dict = read_data_dict_by_client(dataset_list, fold_idx)
        num_of_speakers, speaker_list = len(train_speaker_dict), list(set(train_speaker_dict.keys()))

        #合并验证集
        all_data = []
        all_labels = []
        all_genders = []
        all_dataset = []

        # 遍历每一个键值对
        for key, entry in val_speaker_dict.items():
            # 将data列表扩展到all_data中
            all_data.extend(entry['data'])
            # 将label ndarray追加到all_labels列表中
            all_labels.append(entry['label'])
            # 将gender字符串追加到all_genders中
            all_genders.append(entry['gender'])
            all_dataset.extend(entry['dataset'])

        # 将all_labels中的ndarray合并成一个大的ndarray
        combined_labels = np.concatenate(all_labels)

        # 创建最终的字典
        combined_val_speaker = {
            'data': all_data,
            'label': combined_labels,
            'gender': all_genders,
            'dataset': all_dataset
        }
        
        # Define the model
        seed_everything(8, workers=True)
        torch.manual_seed(8)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if args.pretrain_model == "wav2vec2_0":
            # Wav2vec2_0 Wrapper
            global_model = Wav2VecWrapper(args).to(device)
            # shallow_model = ShallowModel(args, num_shallow_layers=6).to(device)
            # deep_model = DeepModel(args, hidden_dim=256, output_class_num=4, num_shallow_layers=6).to(device)
        elif args.pretrain_model == "wavlm_plus":
            # WavLM Plus Wrapper
            global_model = WavLMWrapper(args).to(device)
        elif args.pretrain_model in ["whisper_tiny", "whisper_base", "whisper_small", "whisper_medium", "whisper_large"]:
            # Whisper Plus Wrapper
            global_model = WhisperWrapper(args).to(device)

        global_weights = global_model.state_dict()
        # global_weights = deep_model.state_dict()

        # 深拷贝模型的方法
        def clone_model(model):
            # 创建新模型实例
            # new_model = Wav2VecWrapper(args).to(device)
            if args.pretrain_model == "wav2vec2_0":
                # Wav2vec2_0 Wrapper
                new_model = Wav2VecWrapper(args).to(device)
                # new_model = ShallowModel(args).to(device)
                # new_model = DeepModel(args).to(device)
            elif args.pretrain_model == "wavlm_plus":
                # WavLM Plus Wrapper
                new_model = WavLMWrapper(args).to(device)
            elif args.pretrain_model in ["whisper_tiny", "whisper_base", "whisper_small", "whisper_medium",
                                         "whisper_large"]:
                # Whisper Plus Wrapper
                new_model = WhisperWrapper(args).to(device)
            # 使用 state_dict 和 load_state_dict 进行参数复制
            new_model.load_state_dict(copy.deepcopy(model.state_dict()))
            return new_model

        # #chenhj
        # # Read trainable params
        # model_parameters = list(filter(lambda p: p.requires_grad, global_model.parameters()))
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # logging.info(f'Trainable params size: {params / (1e6):.2f} M')

        idxs_speakers_epoch = list()
        for epoch in range(int(args.num_epochs)):
            # we choose 20% of clients in training
            np.random.seed(epoch)
            idxs_speakers_epoch.append(np.random.choice(range(num_of_speakers), int(0.5 * num_of_speakers), replace=False))
            # idxs_speakers_epoch.append(np.random.choice(range(num_of_speakers), int(1.0 * num_of_speakers), replace=False))
            
        # log saving path
        seed_everything(8, workers=True)
        torch.manual_seed(8)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        dataloader_dict = dict()

        model_result_path = Path(args.save_dir).joinpath('tmp_model_params1', args.model_type, args.pred, args.feature_type, args.dataset, model_setting_str, save_row_str)
        model_result_csv_path = Path(os.path.realpath(__file__)).parents[1].joinpath('results', args.pred, args.model_type, args.feature_type, model_setting_str)
        Path.mkdir(model_result_path, parents=True, exist_ok=True)
        Path.mkdir(model_result_csv_path, parents=True, exist_ok=True)

        log_path = Path.joinpath(model_result_path, 'log')
        if log_path.exists(): shutil.rmtree(log_path)
        Path.mkdir(log_path, parents=True, exist_ok=True)
        
        # test loader
        dataset_test = DatasetGenerator01(test_speaker_dict, is_train=False)
        test_dataloaders = DataLoader(dataset_test, batch_size=20, num_workers=0, shuffle=False, collate_fn=collate_fn01, drop_last=False)

        # Training steps
        result_dict, best_score = {}, 0
        for epoch in range(int(args.num_epochs)):
            # we choose 10% of clients in training
            idxs_speakers = idxs_speakers_epoch[epoch]
            
            # define list varibles that saves the weights, loss, num_sample, etc.
            local_updates, local_losses, local_num_sampels = [], [], []
            gradient_hist_dict = {}

            # 1. Local training, return weights in fed_avg, return gradients in fed_sgd
            split_arrays = np.array_split(idxs_speakers, 8)
            for i, arr in enumerate(split_arrays):
                speake_id_list = []
                # print(f"第{i + 1}个ndarray包含的元素:")
                # print(arr)
                data = []
                label = []
                gender = []
                dataset = []
                source_id = []
                for idx in arr:
                    speaker_id = speaker_list[idx]
                    speake_id_list.append(speaker_id)
                    data.extend(train_speaker_dict[speaker_id]['data'])
                    label.append(train_speaker_dict[speaker_id]['label'])
                    gender.append(train_speaker_dict[speaker_id]['gender'])
                    dataset.extend(train_speaker_dict[speaker_id]['dataset'])
                    source_id.append([speaker_id] * len(train_speaker_dict[speaker_id]['data']))
                combined_labels = np.concatenate(label)
                combined_source_id = np.concatenate(source_id)
                client_id = 'client_' + str(i + 1)
                combined_dict = {
                    'data': data,
                    'label': combined_labels,
                    'gender': gender,
                    'dataset': dataset,
                    'source_id': combined_source_id
                }

                dataset_train = DatasetGenerator(combined_dict, is_train=True)
                train_dataloaders = DataLoader(dataset_train, batch_size=20, num_workers=0, shuffle=True, collate_fn=collate_fn, drop_last=False)

                # 1.1 Local training
                trainer = local_trainer(args, device, args.model_type, train_dataloaders)
                # read shared updates: parameters in fed_avg and gradients for fed_sgd
                if args.model_type == 'fed_avg':
                    # #chenhj
                    local_update_list, local_update, source_id_result_dict, train_result = trainer.update_weights(model=clone_model(global_model))
                else:
                    local_update, train_result = trainer.update_gradients(model=copy.deepcopy(global_model))
                local_updates.append(copy.deepcopy(local_update))

                # read params to save
                local_losses.append(train_result['loss'])
                local_num_sampels.append(train_result['num_samples'])

                print("agent sample size: " + str(train_result['num_samples']))
                
                # 1.2 calculate and save the raw gradients or pseudo gradients

                if args.model_type == 'fed_avg':

                    original_model = get_trainable_params(clone_model(global_model))

                    for source_id, update_params_dict in local_update_list.items():
                        gradients = []
                        local_update_per_epoch = int(source_id_result_dict[source_id]['num_samples'] / int(args.batch_size)) + 1
                        for key in original_model:
                            original_params = original_model[key].detach().clone().cpu().numpy()
                            update_params = update_params_dict[key].detach().clone().cpu().numpy()

                            # calculate 'fake' gradients
                            tmp_gradients = (original_params - update_params)/(float(args.learning_rate)*local_update_per_epoch*int(args.local_epochs))
                            gradients.append(tmp_gradients)
                            del tmp_gradients, original_params, update_params

                        gradient_hist_dict[source_id] = {}
                        gradient_hist_dict[source_id]['gradient'] = gradients
                        gradient_hist_dict[source_id]['gender'] = train_speaker_dict[source_id]['gender']

                else:
                    for g_idx in range(len(local_update)):
                        gradients.append(local_update[g_idx].cpu().numpy())
                
                # 1.3 save the attack features
                # tmp_key = list(train_speaker_dict[speaker_id].keys())[0]
                del trainer

            # 1.4 dump the gradients for the later usage
            f = open(str(model_result_path.joinpath('gradient_hist_'+str(epoch)+'.pkl')), "wb")
            pickle.dump(gradient_hist_dict, f)
            f.close()
            
            # 2. global model updates
            total_num_samples = np.sum(local_num_sampels)
            if args.model_type == 'fed_avg':
                # 2.1 average global weights
                # global_weights = average_weights(local_updates, local_num_sampels)
                average_w = average_weights(local_updates, local_num_sampels)
                for key, value in average_w.items():
                    if key in global_weights:
                        global_weights[key] = value

            else:
                # 2.1 average global gradients
                global_gradients = average_gradients(local_updates, local_num_sampels)
                # 2.2 update global weights
                global_weights = copy.deepcopy(global_model.state_dict())
                global_weights_keys = list(global_weights.keys())
                
                for key_idx in range(len(global_weights_keys)):
                    key = global_weights_keys[key_idx]
                    global_weights[key] -= float(args.learning_rate)*global_gradients[key_idx].to(device)
                del global_gradients
            
            # 2.3 load new global weights
            global_model.load_state_dict(global_weights)

            # 3. Calculate avg validation accuracy/uar over all selected users at every epoch
            validation_acc, validation_uar, validation_loss, local_num_sampels = [], [], [], []
            # 3.1 Iterate each client at the current global round, calculate the performance
            # for idx in range(num_of_speakers):
            # speaker_id = speaker_list[idx]
            dataset_validation = DatasetGenerator01(combined_val_speaker, is_train=False)
            val_dataloaders = DataLoader(dataset_validation, batch_size=20, num_workers=0, shuffle=False, collate_fn=collate_fn01, drop_last=False)

            trainer = local_trainer(args, device, args.model_type, val_dataloaders)
            # # chenhj
            local_val_result = trainer.inference(clone_model(global_model), process="validate")

            # save validation accuracy, uar, and loss
            local_num_sampels.append(local_val_result['num_samples'])
            validation_acc.append(local_val_result['acc'])
            validation_uar.append(local_val_result['uar'])
            validation_loss.append(local_val_result['loss'])
            del val_dataloaders, trainer
            
            # 3.2 Re-Calculate weigted performance scores
            validate_result = {}
            weighted_acc, weighted_rec = 0, 0
            total_num_samples = np.sum(local_num_sampels)
            for acc_idx in range(len(validation_acc)):
                weighted_acc += validation_acc[acc_idx] * (local_num_sampels[acc_idx] / total_num_samples)
                weighted_rec += validation_uar[acc_idx] * (local_num_sampels[acc_idx] / total_num_samples)
            validate_result['acc'], validate_result['uar'] = weighted_acc, weighted_rec
            validate_result['loss'] = np.mean(validation_loss)

            print('| Global Round validation : {} | \tacc: {:.2f}% | \tuar: {:.2f}% | \tLoss: {:.6f}\n'.format(
                        epoch, weighted_acc*100, weighted_rec*100, validate_result['loss']))
            
            # 4. Perform the test on holdout set
            trainer = local_trainer(args, device, args.model_type, test_dataloaders)
            # # chenhj

            test_result = trainer.inference(clone_model(global_model), process="test")
            
            # 5. Save the results for later
            result_dict[epoch] = {}
            result_dict[epoch]['train'] = {}
            result_dict[epoch]['train']['loss'] = sum(local_losses) / len(local_losses)
            result_dict[epoch]['validate'] = validate_result
            result_dict[epoch]['test'] = test_result
            
            if epoch == 0: best_epoch, best_val_dict, best_test_dict = 0, validate_result, test_result
            if validate_result['uar'] > best_val_dict['uar']:
                # Save best model and training history
                best_epoch, best_val_dict, best_test_dict = epoch, validate_result, test_result
                torch.save(deepcopy(global_model.state_dict()), str(model_result_path.joinpath('model.pt')))
            
            if epoch > 10:
                # log results
                print('best epoch %d, best final acc %.2f, best val acc %.2f' % (best_epoch, best_test_dict['acc']*100, best_val_dict['acc']*100))
                print('best epoch %d, best final rec %.2f, best val rec %.2f' % (best_epoch, best_test_dict['uar']*100, best_val_dict['uar']*100))
                print(best_test_dict['conf'])
        
        # Performance save code
        row_df = save_result(save_row_str, best_test_dict['acc'], best_test_dict['uar'], best_epoch, args.dataset)
        save_result_df = pd.concat([save_result_df, row_df])
        save_result_df.to_csv(str(model_result_csv_path.joinpath('private_'+ str(args.dataset) + '.csv')))
        
        f = open(str(model_result_path.joinpath('results.pkl')), "wb")
        pickle.dump(result_dict, f)
        f.close()

    # Calculate the average of the 5-fold experiments
    tmp_df = save_result_df.loc[save_result_df['dataset'] == args.dataset]
    row_df = save_result('average', np.mean(tmp_df['acc']), np.mean(tmp_df['uar']), best_epoch, args.dataset)
    save_result_df = pd.concat([save_result_df, row_df])
    save_result_df.to_csv(str(model_result_csv_path.joinpath('private_'+ str(args.dataset) + '.csv')))
