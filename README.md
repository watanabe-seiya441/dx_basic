# 建築AIコード
建物AIのモデル作成のための基本のコードです。

## python 仮想環境を使う

0. uvを使うためにgpu2の環境変数を適用する。
```
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
```

1. uv環境の作成
```
cd ~/dx_basic
uv sync
```
2. uv環境に入る
```
source .venv/bin/activate
```
ここの部分は.bashrcとかのエイリアスで高速化できる。  
例 ~/.bashrcに以下を書き込んで`source ~/.bashrc`することで`dx_basic`で仮想環境に入ることができる。
```
alias dx_basic="source $HOME/dx_basic/.venv/bin/activate"
```
仮想環境から抜けるのは`exit`で    
これから環境に入る時は`dx_basic`で


## データや結果は基本的に/mnt/data-raid/{username}/dx/に保存する

```
cd /mnt/data-raid/{username}/dx/
mkdir data result
cd /home/{username}/dx_basic/
ln -s /mnt/data-raid/{username}/dx/data data
ln -s /mnt/data-raid/{username}/dx/result result
```


`dataset0a`などのdatasetディレクトリは/mnt/data-raid/{username}/dx/data/の下に置く

## ハイパーパラメータ設定
`/home/{ユーザ名}/dx_basic/config/config.toml`にパラメータを保存する。

```
[model]
net = "vgg19_bn"
pretrained = true
transfer = false

[training]
lr = 0.001
momentum = 0.9
num_epochs = 5
batch_size = 10

[device]
nvidia = 0

[data]
num_val = 8
which_data = "datasetA0a_run12341w2w_run53w"
train_data = "train"
test_data = "test"
```

## csv について
{project_root}/result/[datasetの名前]/[保存日時とパラメータのセット名]/に保存される history.csv の内容:

| 列番号 | 説明                                                                                     |
|--------|------------------------------------------------------------------------------------------|
| 0      | 世代                                                                                     |
| 1      | 訓練データの損失 (`avg_train_loss`)                                                      |
| 2      | 訓練データの精度 (`train_acc`)                                                           |
| 3      | 検証データの損失 (`avg_val_loss`)                                                        |
| 4      | 検証データの4ミス精度 (`val_acc[0]`) e.g., True: `0000` → Predicted: `1111`              |
| 5      | 検証データの3ミス精度 (`val_acc[1]`) e.g., True: `0000` → Predicted: `1110`              |
| 6      | 検証データの2ミス精度 (`val_acc[2]`) e.g., True: `0000` → Predicted: `1100`              |
| 7      | 検証データの1ミス精度 (`val_acc[3]`) e.g., True: `0000` → Predicted: `1000`              |
| 8      | 検証データの0ミス精度 (`val_acc[4]`) e.g., True: `0000` → Predicted: `0000`              |
| 9      | 訓練データのバランス精度 (`balanced_acc_dict["train_BA"]`)                               |
| 10     | 検証データのバランス精度 (`balanced_acc_dict["test_BA"]`)                                |


## スクリプトの概要
- main.py
    - data/とconfig/config.tomlを参照して学習を行う
    - 結果は`result/[datasetの名前]/[main.pyを実行した時刻とconfig/config.tomlに基づくディレクトリ名]/`に保存
    - 例えば、`result/dataset0a/2024-11-21_07-45-22_resnet50-0.001-0.9-25-10/`に保存

## プログラム実行方法(基本)
```
# 通常の実行
python main.py

# バックグラウンド実行
nohup python main.py &
```
nohup.outに出力される。


## ディレクトリ構成
- config/ 
    - config.tomlで学習パラメータを設定

- mylib/
    - main.pyに使う自作モジュールを格納

    - evaluate.py
        - history変数より、(x:epoch、y:損失)、(x:epoch、y:精度)という2つのグラフを可視化
    - fit.py
        - モデル学習部分
    - make_confusion_matrix.py
        - 混同行列を作成
    - save_history_to_csv.py
        - history.csvを作成
    - torch_seed.py
        - seed値を定める

