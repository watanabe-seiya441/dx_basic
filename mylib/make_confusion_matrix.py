import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(
    cm, #confusion matrix
    classes,
    normalize=False,
    title="Confusion Matrix",
    cmap=plt.cm.Blues,
    save_path=None
):
    len_classes = len(classes)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # e.g. [[5, 3], [2, 6]] -> [5/8, 3/8], [2/8, 6/8]]
        plt.rcParams.update({"font.size": 8})

    plt.clf() # 既存のプロットをクリア
    plt.imshow(cm, interpolation="nearest", cmap=cmap) # 画像を表示
    plt.title(title, fontsize=20)

    ## メモリの文字サイズを調整
    cbar = plt.colorbar()  # カラーバーを追加し、Colorbarクラスをcbarに代入 c: color
    cbar.ax.tick_params(labelsize=12)

    ## クラスラベルをx軸とy軸に設定し、目盛りの位置とフォントサイズを調整
    tick_marks = np.arange(len_classes)
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 # 文字色を切り替えるための閾値

    ## この部分の意味
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black", #threshを超えたら白色
            fontsize=7
        )
    #コメントで教えて

    plt.tight_layout()
    plt.xlabel("Predicted label", fontsize=14)
    plt.ylabel("True label", fontsize=14)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        plt.savefig(save_path)
        

def make_confusion_matrix(device, epoch, classes, test_loader, save_dir, net):
    y_preds = []
    y_tests = []

    with torch.no_grad():
        for inputs_test, labels_test in test_loader:
            
            # GPUへ転送
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            outputs_test = net(inputs_test) # outputs_test.shape = [test_batch_size, len_classes]

            # 予測ラベルを取得
            _, labels_predicted = torch.max(outputs_test, 1)

            # リストに追加
            y_preds.extend(labels_predicted.cpu().tolist())
            y_tests.extend(labels_test.cpu().tolist())

    # 混同行列を作成
    dx_confusion_matrix = confusion_matrix(y_tests, y_preds)
    
    # 混同行列が空の場合の処理
    if dx_confusion_matrix.size == 0:
        print("混同行列が空です。予測や正解ラベルを確認してください。")
        return

    # 混同行列を保存(実数値)
    save_confusion_matrix(
        dx_confusion_matrix,
        classes,
        normalize=False,
        title=f"Confusion Matrix at {epoch}",
        cmap=plt.cm.Reds,
        save_path=os.path.join(
            save_dir,
            "confusion_matrix",
            f"cm_count_{epoch}.png",
        )
    )

    print(f"混同行列を保存しました: {save_dir}/confusion_matrix/cm_count_{epoch}.png")

    ### 予測ラベルの確率値のconfusion matrix
    save_confusion_matrix(
        dx_confusion_matrix,
        classes,
        normalize=True,
        title=f"Confusion Matrix at {epoch}",
        cmap=plt.cm.Reds,
        save_path=os.path.join(
            save_dir,
            "confusion_matrix",
            f"cm_count_{epoch}_norm.png",
        )
    )

    print(f"混同行列を保存しました: {save_dir}/confusion_matrix/cm_count_{epoch}_norm.png")


# スクリプトとして実行された場合のみ、以下のコードを実行
if __name__ == "__main__":
    import toml
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # プロジェクトルートディレクトリを取得 /home/watanabe/dx/develop/を想定
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    with open(f"{project_root}/example_result/config.toml", "r") as f:
        config = toml.load(f)

    which_data = config["data"]["which_data"] # e.g. dataset0a
    test_data = config["data"]["test_data"] # e.g. test
    test_dir = os.path.join(project_root, "data", which_data, test_data)
    classes = sorted([name for name in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, name))])
    net = torch.load(f"{project_root}/example_result/epoch25.pth")
    batch_size = config["training"]["batch_size"]
    device = config['device']['nvidia']
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    num_epochs = config["training"]["num_epochs"]

    test_transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False
    )

    # 混同行列保存用ディレクトリ作成
    os.makedirs(f"{project_root}/example_result/confusion_matrix", exist_ok=True)

    save_dir = f"{project_root}/example_result"

    # 実行例
    make_confusion_matrix(
        device=device,
        epoch=num_epochs,
        classes=classes,
        test_loader=test_loader,
        save_dir=save_dir,
        net=net
    )
