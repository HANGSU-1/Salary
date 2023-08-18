import pandas as pd
import torch
import numpy as np

see = pd.read_csv(r"F:\Salary_dataset.csv")
see.to_csv("Salary_dataset.csv")
#print(pd.read_csv("Salary_dataset.csv"))
def read_data():
    # データの読み込み
    tips_csv = pd.read_csv("Salary_dataset.csv", index_col=0, header=0)
    # 読み込んだデータの確認
    # print(tips_csv.head())

    # NNで処理できるようにデータを変換
    tips_data = tips_csv

    return tips_data


# データをPyTorchでの学習に利用できる形式に変換
def create_dataset_from_dataframe(tips_data, target_tag="Salary"):
    # "tip"の列を目的にする
    target = torch.tensor(tips_data[target_tag].values, dtype=torch.float32).reshape(-1, 1)
    # "tip"以外の列を入力にする
    input = torch.tensor(tips_data["YearsExperience"].values, dtype=torch.float32).reshape(-1,1)
    return input, target


# 4層順方向ニューラルネットワークモデルの定義
class FourLayerNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, output_size)
        self.do = torch.nn.Dropout()

    def forward(self, x):
        h1 = torch.relu(self.l1(x))
        o = self.l3(h1)
        return o


def train_model(nn_model, input, target):
    # データセットの作成
    tips_dataset = torch.utils.data.TensorDataset(input, target)
    # バッチサイズ=25として学習用データローダを作成
    train_loader = torch.utils.data.DataLoader(tips_dataset, batch_size=5, shuffle=True)

    # オプティマイザ
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.0000000001, momentum=0.9)

    # データセット全体に対して10000回学習
    for epoch in range(1000000):
        # バッチごとに学習する
        for x, y_hat in train_loader:
            y = nn_model(x)
            loss = torch.nn.functional.mse_loss(y, y_hat)#二乗誤差で学習

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 1000回に1回テストして誤差を表示
        if epoch % 10000 == 0:
            with torch.inference_mode():  # 推論モード（学習しない）
                y = nn_model(input)
                loss = torch.nn.functional.mse_loss(y, target)
                print(epoch, loss)


# データの準備
tips_data = read_data()
input, target = create_dataset_from_dataframe(tips_data)

# NNのオブジェクトを作成
nn_model = FourLayerNN(input.shape[1], 30, 1)
train_model(nn_model, input, target)

# 学習後のモデルの保存
# torch.save(nn_model.state_dict(), "nn_model.pth")

# 学習後のモデルのテスト
test_data = torch.tensor(
    [
        [
            [1],[3],[6]#yearsExperience
        ]
    ],
    dtype=torch.float32,
)
with torch.inference_mode():  # 推論モード（学習しない）
    print(nn_model(test_data))
