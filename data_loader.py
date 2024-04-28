import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import torch.nn as nn
X_label_encoder = LabelEncoder()
Y_label_encoder = LabelEncoder()


class CustomDataset(Dataset):
    def __init__(self, args, is_train=True):
        self.args = args
        self.is_train = is_train
        self.load_data()
        self.modify()
        self.torch_form()


    def load_data(self):
        s = self.args.train_subject[0]
        if self.args.phase == 'train':
            self.X = pd.read_csv(f"./data/S{s:02}_train_X.csv")
            self.y = pd.read_csv(f"./data/S{s:02}_train_Y.csv")
        else:
            self.X = pd.read_csv(f"./data/S{s:02}_test_X.csv")
            self.y = pd.read_csv(f"./data/S{s:02}_test_Y.csv")
        # if len(self.X.shape) <= 3:
        #     self.X = np.expand_dims(self.X, axis=1)


    def modify(self):
        global X_label_encoder, Y_label_encoder
        # 날짜를 파싱하고 연, 월, 일 데이터로 변환

        self.X['대여일자'] = pd.to_datetime(self.X['대여일자'], format='%Y-%m-%d')

        self.X['year'] = self.X['대여일자'].dt.year
        self.X['month'] = self.X['대여일자'].dt.month
        self.X['대여일자'] = self.X['대여일자'].dt.day
        self.X.drop(['year', 'month', '대여소'], axis=1, inplace=True)
        self.X['성별'] = self.X['성별'].fillna('모름')
        self.X['성별'] = self.X['성별'].replace('f', 'F')
        self.X['성별'] = self.X['성별'].replace('m', 'M')
        self.X['연령대'] = self.X['연령대'].fillna('모름')

        # # 범주형 데이터를 숫자로 매핑
        # gender_mapping = {'M': 0, 'F': 1, '모름': -1}
        # age_group_mapping = {'~10대': 0, '20대': 1, '30대':2,'40대':3,'50대': 4, '60대': 5, '70대': 6, '기타':-1, '모름':-2}
        # self.X['성별'] = self.X['성별'].map(gender_mapping)
        # self.X['연령대'] = self.X['연령대'].map(age_group_mapping)
        # 범주형 데이터 인코딩
        categorical_columns = self.X.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            self.X[column] = X_label_encoder.fit_transform(self.X[column])

        # 정규화
        self.X['운동량'] = (self.X['운동량'] - self.X['운동량'].mean()) / self.X['운동량'].std()
        self.X['탄소량'] = (self.X['탄소량'] - self.X['탄소량'].mean()) / self.X['탄소량'].std()
        self.X['이동거리(M)'] = (self.X['이동거리(M)'] - self.X['이동거리(M)'].mean()) / self.X['이동거리(M)'].std()
        self.X['이용시간(분)'] = (self.X['이용시간(분)'] - self.X['이용시간(분)'].mean()) / self.X['이용시간(분)'].std()


        self.y['대여구분코드'] = Y_label_encoder.fit_transform(self.y['대여구분코드'])

    def torch_form(self):
        print(self.X.dtypes)
        for column in self.X.columns:
            self.X[column] = pd.Categorical(self.X[column])
            self.X[column] = self.X[column].cat.codes

        # self.X = self.X.to_numpy()
        # 텐서 변환
        categorical_data = torch.tensor(self.X[['대여소번호', '성별', '연령대']].values, dtype=torch.long)
        numerical_data = torch.tensor(self.X[['운동량', '탄소량','이동거리(M)', '이용시간(분)']].values, dtype=torch.float)
        combined_data = torch.cat((categorical_data, numerical_data), dim=1)
        # 데이터셋 생성
        # dataset = TensorDataset(combined_data)
        self.X = combined_data
        for column in self.y.columns:
            self.y[column] = pd.Categorical(self.y[column])
            self.y[column] = self.y[column].cat.codes
        self.y = torch.tensor(self.y.values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]
        sample = [X, y]
        return sample


def data_loader(args):
    print("[Load data]")
    # Load train data

    args.phase = "train"

    trainset = CustomDataset(args,is_train=True)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)

    # Load val data
    args.phase = "val"
    valset = CustomDataset(args,is_train=False)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Print
    print(f"train_set size: {len(train_loader)}")
    print(f"val_set size: {len(val_loader)}")
    print("")
    return train_loader, val_loader
