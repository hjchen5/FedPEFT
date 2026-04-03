import pandas as pd
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import copy, pdb, time, warnings, torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')
import loralib as lora
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import OrderedDict
from collections import defaultdict

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

#chenhj 该函数会遍历模型中的所有参数，并仅返回那些 requires_grad=True 的参数
def get_trainable_params(model):
    """
    Returns a state dict containing only the parameters that require gradients.
    """
    trainable_state_dict = OrderedDict((name, param.clone().detach()) for name, param in model.named_parameters() if param.requires_grad)
    # trainable_state_dict = OrderedDict((k, v) for k, v in model.state_dict().items() if v.requires_grad)
    # trainable_state_dict = {name: param for name, param in model.state_dict().items() if model.get_parameter(name).requires_grad}
    return trainable_state_dict

def average_weights(w, num_samples_list):
    """
    Returns the average of the weights.
    """
    total_num_samples = np.sum(num_samples_list)
    w_avg = copy.deepcopy(w[0])

    for key in w_avg.keys():
        w_avg[key] = w[0][key]*(num_samples_list[0]/total_num_samples)
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += torch.div(w[i][key]*num_samples_list[i], total_num_samples)
    return w_avg


def average_gradients(g, num_samples_list):
    """
    Returns the average of the gradients.
    """
    total_num_samples = np.sum(num_samples_list)
    g_avg = copy.deepcopy(g[0])
    
    for layer_idx in range(len(g[0])):
        g_avg[layer_idx] = g[0][layer_idx] * (num_samples_list[0]/total_num_samples)
    for layer_idx in range(len(g[0])):
        for client_idx in range(1, len(g)):
            g_avg[layer_idx] += torch.div(g[client_idx][layer_idx]*num_samples_list[client_idx], total_num_samples)
    return g_avg


def result_summary(step_outputs):
    loss_list, y_true, y_pred = [], [], []
    for step in range(len(step_outputs)):
        for idx in range(len(step_outputs[step]['pred'])):
            y_true.append(step_outputs[step]['truth'][idx])
            y_pred.append(step_outputs[step]['pred'][idx])
        loss_list.append(step_outputs[step]['loss'])

    result_dict = {}
    acc_score = accuracy_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred, average='macro')
    confusion_matrix_arr = np.round(confusion_matrix(y_true, y_pred, normalize='true')*100, decimals=2)
    
    result_dict['acc'] = acc_score
    result_dict['uar'] = rec_score
    result_dict['conf'] = confusion_matrix_arr
    result_dict['loss'] = np.mean(loss_list)
    result_dict['num_samples'] = len(y_pred)
    return result_dict


class local_trainer(object):
    def __init__(self, args, device, model_type, dataloader):
        self.args = args
        self.device = device
        # self.optimizer = optimizer
        # self.scheduler = scheduler
        self.model_type = model_type
        self.dataloader = dataloader
        # self.weights = weights
        
    def update_weights(self, model):
        # # Set mode to train model
        # model.train()
        # step_outputs = []
        # criterion = nn.CrossEntropyLoss().to(self.device)
        # # # Define optimizer
        # optimizer = torch.optim.Adam(
        #     list(filter(lambda p: p.requires_grad, model.parameters())),
        #     lr=float(self.args.learning_rate),
        #     weight_decay=1e-4,
        #     betas=(0.9, 0.98)
        # )
        # # Define scheduler, patient = 5, minimum learning rate 5e-5
        # scheduler = ReduceLROnPlateau(
        #     optimizer, mode='min', patience=5, factor=0.5, verbose=True, min_lr=5e-5
        # )
        #
        # trainable_params_list = defaultdict(list)
        # for iter in range(int(self.args.local_epochs)):
        #     for batch_idx, batch_data in enumerate(self.dataloader):
        #         model.zero_grad()
        #         optimizer.zero_grad()
        #         x, y, length, source_id = batch_data
        #         x, y, length = x.to(self.device), y.to(self.device), length.to(self.device)
        #         # weight1 = model.state_dict()
        #         logits = model(x, length=length)
        #         loss = criterion(logits, y)
        #         loss.backward()
        #         optimizer.step()
        #
        #         # if (batch_idx % 10 == 0 and batch_idx != 0) or batch_idx == len(self.dataloader) - 1:
        #         #     logging.info(f'Current Train LR: {scheduler.optimizer.param_groups[0]["lr"]}')
        #
        #         predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        #         pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
        #         truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
        #         step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})
        #
        #         # 保存当前 source_id 对应的模型参数
        #         current_params = get_trainable_params(model)
        #         for sid in source_id:
        #             trainable_params_list[sid].append(current_params)
        #
        #     # 更新学习率
        #     scheduler.step(loss)
        #
        # result_dict = result_summary(step_outputs)
        # # return model.state_dict(), result_dict
        # # return lora.lora_state_dict(model), result_dict
        # trainable_params = get_trainable_params(model)
        # # 将 trainable_params_list 转换为所需的形式
        # final_trainable_params_list = {k: v[-1] for k, v in trainable_params_list.items()}
        #
        # # 将 trainable_params_list 转换为所需的形式
        # return final_trainable_params_list, trainable_params, result_dict

        model.train()
        criterion = nn.CrossEntropyLoss().to(self.device)

        optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, model.parameters())),
            lr=float(self.args.learning_rate),
            weight_decay=1e-4,
            betas=(0.9, 0.98)
        )

        # 保持所有 source_id 对应的模型参数
        trainable_params_list = defaultdict(list)
        source_id_result_dict = defaultdict(list)
        step_outputs = []

        for iter in range(int(self.args.local_epochs)):
            source_id_batches = defaultdict(list)

            # 先将所有批次按 source_id 分组
            for batch_data in self.dataloader:
                x, y, length, source_id = batch_data
                for sid in source_id:
                    source_id_batches[sid.item()].append(batch_data)

            # 然后按每个 source_id 进行训练
            for sid, batches in source_id_batches.items():
                source_id_step_outputs = []
                for batch_data in batches:
                    model.zero_grad()
                    optimizer.zero_grad()
                    x, y, length, source_id = batch_data
                    x, y, length = x.to(self.device), y.to(self.device), length.to(self.device)
                    logits = model(x, length=length)
                    # logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

                    predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
                    pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
                    truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
                    source_id_step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})
                    step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})

                # 保存当前 source_id 对应的模型参数
                current_params = get_trainable_params(model)
                trainable_params_list[sid].append(current_params)
                source_id_result_dict[sid] = result_summary(source_id_step_outputs)

        result_dict = result_summary(step_outputs)
        trainable_params = get_trainable_params(model)
        final_trainable_params_list = {k: v[-1] for k, v in trainable_params_list.items()}

        return final_trainable_params_list, trainable_params, source_id_result_dict, result_dict


    def update_gradients(self, model):
        # Set mode to train model
        model.train()
        step_outputs = []
        for batch_idx, batch_data in enumerate(self.dataloader):
            if batch_idx == 0:
                x, y, dataset = batch_data
                x, y = x.to(self.device), y.to(self.device)

                model.zero_grad()
                logits = model(x.float())
                loss = self.criterion(logits, y)
                loss.backward()
                grads = [param.grad.detach().clone() for param in model.parameters()]

                predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
                pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
                truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
                step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})
        result_dict = result_summary(step_outputs)
        return grads, result_dict

    def inference(self, model, process):
        model.eval()
        step_outputs = []
        criterion = nn.CrossEntropyLoss().to(self.device)

        # # Define optimizer
        # optimizer = torch.optim.Adam(
        #     list(filter(lambda p: p.requires_grad, model.parameters())),
        #     lr=float(self.args.learning_rate),
        #     weight_decay=1e-4,
        #     betas=(0.9, 0.98)
        # )

        # # Define scheduler, patient = 5, minimum learning rate 5e-5
        # scheduler = ReduceLROnPlateau(
        #     optimizer, mode='min', patience=5, factor=0.5, verbose=True, min_lr=5e-5
        # )

        # with torch.no_grad():
        for batch_idx, batch_data in enumerate(self.dataloader):
            x, y, length = batch_data
            x, y, length = x.to(self.device), y.to(self.device), length.to(self.device)
            # weights1 = model.state_dict()
            logits = model(x, length=length)
            loss = criterion(logits, y)

            predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
            pred_list = [predictions[pred_idx] for pred_idx in range(len(predictions))]
            truth_list = [y.detach().cpu().numpy()[pred_idx] for pred_idx in range(len(predictions))]
            step_outputs.append({'loss': loss.item(), 'pred': pred_list, 'truth': truth_list})
        result_dict = result_summary(step_outputs)
        # if process=="validate": scheduler.step(result_dict["loss"])
        return result_dict

def noise_add(noise_scale, w, device):
    w_noise = copy.deepcopy(w)
    for i in w.keys():
        noise = np.random.normal(0, noise_scale, w[i].size())
        noise = torch.from_numpy(noise).float().to(device)
        w_noise[i] = w_noise[i] + noise
    return w_noise
