## main.py

import os
import pathlib
import shutil
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch # 2.5.1+cu124
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import toml

from mylib.fit import fit
from mylib.save_history_to_csv import save_history_to_csv
from mylib.evaluate import evaluate_history


plt.rcParams["font.size"] = 18
plt.tight_layout()

with open("config/config.toml", "r") as f:
    config = toml.load(f)
print("configを読み取りました。")

# note: config_netとnetを分ける必要が生じた。どこかで改善したいかもしれない。
config_net = config["model"]["net"]
pretrained = config["model"]["pretrained"]
transfer = config["model"]["transfer"]

lr = config["training"]["lr"]
momentum = config["training"]["momentum"]
num_epochs = config["training"]["num_epochs"]
batch_size = config["training"]["batch_size"]


start_time = time.time()


nvidia = config['device']['nvidia']
device = torch.device(f"cuda:{nvidia}" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("GPUが使えます。")
else:
    print("CPUを使います。")

num_val = config["data"]["num_val"]
which_data = config["data"]["which_data"] # e.g. "dataset0a"
train_data = config["data"]["train_data"] # e.g. "train"
test_data = config["data"]["test_data"] # e.g. "test"

which_data = config["data"]["which_data"]
root_dir = os.getcwd()
train_dir = os.path.join(root_dir, "data", which_data, train_data)
test_dir = os.path.join(root_dir, "data", which_data, test_data)

# 現在時刻を得る
now = datetime.now()

Date = now.strftime("%Y-%m-%d") 
Time = now.strftime("%H-%M-%S") 


when = f"{Date}_{Time}_{config_net}-{lr}-{momentum}-{num_epochs}-{batch_size}"
# e.g. 2024-11-26_14-47-43_vgg19_bn-0.001-0.9-25-10

### 実行に失敗した時にディレクトリを削除するためのtry-except文 ###
try:
    print("\033[94mtryブロックを開始\033[0m")

    # 実行結果の保存用ディレクトリ
    save_dir = os.path.join(root_dir, "result", which_data, when) 
    # e.g. result/dataset0a/2021-09-01_12-34-56_resnet18-0.001-0.9-100-32
    os.makedirs(save_dir, exist_ok=True)

    # 実行時tomlを保存
    shutil.copy(src=os.path.join(root_dir, "config", "config.toml"),
                dst=save_dir)
    print("config.tomlを保存しました。")

    test_transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    classes = sorted([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))])
    # e.g. ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111']

    train_data = datasets.ImageFolder(train_dir, transform=test_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True 
    ) 
    test_loader = DataLoader(
        test_data, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False
    )

    from torchvision import models
    from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights
    from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet152_Weights
    from torchvision.models import VGG16_BN_Weights, VGG19_BN_Weights

    # config.netに対応するモデル名とモデルクラスの対応を定義した辞書
    # lambda式でpretrainedを判定
    model_mapping = {
        "vit_b_16": lambda: models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None),
        "vit_l_16": lambda: models.vit_l_16(weights=ViT_L_16_Weights.DEFAULT if pretrained else None),
        "resnet18": lambda: models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None),
        "resnet50": lambda: models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None),
        "resnet152": lambda: models.resnet152(weights=ResNet152_Weights.DEFAULT if pretrained else None),
        "vgg16_bn": lambda: models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT if pretrained else None),
        "vgg19_bn": lambda: models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT if pretrained else None),
    }
    
    net = model_mapping[config_net]()

    ## シード値を固定
    from mylib.torch_seed import torch_seed
    torch_seed()

    if transfer:
        for param in net.parameters():
            param.requires_grad = False
    
    len_classes = len(classes)
    if "vit" in config_net:
        fc_in_features = net.heads.head.in_features
        net.heads.head = nn.Linear(fc_in_features, len_classes)
    elif "resnet" in config_net:
        fc_in_features = net.fc.in_features
        net.fc = nn.Linear(fc_in_features, len_classes)
    elif "vgg" in config_net:
        in_features = net.classifier[6].in_features                         
        net.classifier[6] = nn.Linear(in_features, len_classes)  
        net.avgpool = nn.Identity() 

    net = net.to(device)

    criterion = nn.CrossEntropyLoss() 

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # historyは、epoch+1,avg_train_loss,train_acc,avg_val_loss,*val_acc(5個),balanced_acc_dict["train_BA"],balanced_acc_dict["test_BA"]の順
    history = np.zeros((0, 11))

    program_name = sys.argv[0]

    from mylib.fit import fit
    history = fit(
        net,
        optimizer,
        criterion,
        num_epochs,
        classes,
        train_loader,
        test_loader,
        device,
        history,
        program_name,
        save_dir,
        which_data,
        len_classes,
        save_model=True,
        save_cm_ls=True,
    )

    print(f"csvを{save_dir}に保存")
    save_history_to_csv(history, save_dir)    
    
    # 損失、精度の評価を保存
    print(f"損失、精度を{save_dir}/ に保存")
    evaluate_history(history, save_dir, which_data)

    end_time = time.time()
    execution_time = end_time - start_time
    print("実行時間:", execution_time, "秒")

except KeyboardInterrupt:
    print("実行が中断されました。ディレクトリを削除します。")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    print("ディレクトリが削除されました。")

except Exception as e:
    import traceback
    traceback.print_exc() 
    print("エラーが発生しました:", e)
    print("ディレクトリを削除します。")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    print("ディレクトリが削除されました。")

finally:
    print("end")