import os
import pandas as pd
from datasets import Dataset, DatasetDict
import torchaudio

def load_dataset(dataset_path):
    sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    data = []

    for session in sessions:
        wav_path = os.path.join(dataset_path, session, 'sentences', 'wav')
        emo_evaluation_path = os.path.join(dataset_path, session, 'dialog', 'EmoEvaluation')

        # 读取情感标签文件
        for root, _, files in os.walk(emo_evaluation_path):
            for file in files:
                if file.endswith('.txt'):
                    with open(os.path.join(root, file), 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.startswith('['):  # 标签行通常以 [ 开头
                                parts = line.split()
                                start_time = float(parts[0][1:])
                                end_time = float(parts[1][:-1])
                                emotion = parts[2]
                                file_id = parts[3].split('_')[-1] + '.wav'
                                wav_file_path = os.path.join(wav_path, file_id)
                                if os.path.exists(wav_file_path):
                                    data.append({"file_path": wav_file_path, "emotion": emotion})

    # 转换为 pandas DataFrame
    df = pd.DataFrame(data)

    # 根据情感标签进行过滤，只保留主要情感（例如，angry, happy, sad, neutral）
    main_emotions = ['ang', 'hap', 'sad', 'neu']
    df = df[df['emotion'].isin(main_emotions)]

    # 将情感标签转换为数字编码
    emotion_mapping = {emotion: i for i, emotion in enumerate(main_emotions)}
    df['emotion'] = df['emotion'].map(emotion_mapping)

    # 划分训练集和测试集（例如按会话划分，Session 1-4 为训练集，Session 5 为测试集）
    train_sessions = ['Session1', 'Session2', 'Session3', 'Session4']
    test_sessions = ['Session5']

    train_df = df[df['file_path'].str.contains('|'.join(train_sessions))]
    test_df = df[df['file_path'].str.contains('|'.join(test_sessions))]

    # 将 pandas DataFrame 转换为 Hugging Face 的 Dataset 格式
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 将数据集封装到 DatasetDict 中
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    return dataset_dict

# 使用示例
dataset_path = "/home/chenhj/WorkSpaces_python/datasets/iemocap"
iemocap_dataset = load_dataset(dataset_path)
