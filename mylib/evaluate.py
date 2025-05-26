import os

import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np

# history.csvの列の形式は以下を想定している 全て5桁想定
# 0. 世代 (epoch+1)
# 1. 訓練データの損失(avg_train_loss)
# 2. 訓練データの精度(train_acc)
# 3. 検証データの損失(avg_val_loss)
# 4. 検証データの4ミス精度(val_acc[0])
# 5. 検証データの3ミス精度(val_acc[1])
# 6. 検証データの2ミス精度(val_acc[2])
# 7. 検証データの1ミス精度(val_acc[3])
# 8. 検証データの0ミス精度(val_acc[4])
# 9. 訓練データのバランス精度(balanced_acc_dict["train_BA"])
# 10. 検証データのバランス精度(balanced_acc_dict["test_BA"])

# 損失、精度を評価
def evaluate_history(history, save_dir, which_data):

    result_f = open(
        f"{save_dir}/abstract.txt",
        "a",
        newline="\n",
    )

    datalines = [
        f"使用した訓練データ: {which_data}\n\n",
        "検証データの成績\n",
        f"初期状態: 損失={history[0, 3]}, 精度={history[0, 8]} バランス精度: {history[0,10]}\n",
        f"最終状態: 損失={history[-1, 3]}, 精度={history[-1, 8]} バランス精度: {history[-1,10]}\n",
    ]
    result_f.writelines(datalines)
    result_f.close()

    num_epochs = history.shape[0]
    if num_epochs < 10:
        unit = 1
    else:
        unit = num_epochs / 10

    # 損失の推移
    plt.figure(figsize=(9, 8))
    plt.plot(history[:, 0], history[:, 1], "b", label="訓練")
    plt.plot(history[:, 0], history[:, 3], "k", label="検証")
    plt.xticks(np.arange(0, num_epochs + 1, unit))
    plt.xlabel("繰り返し回数")
    plt.ylabel("損失")
    plt.title("学習曲線(損失)")
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{which_data}_loss.png"))
    plt.show()

    # 精度の推移
    plt.figure(figsize=(9, 8))
    plt.plot(history[:, 0], history[:, 2], "b", label="訓練")
    plt.plot(history[:, 0], history[:, 8], "k", label="検証")
    plt.plot(history[:, 0], history[:, 7], "g", label="1miss")
    plt.plot(history[:, 0], history[:, 6], "c", label="2miss")
    plt.plot(history[:, 0], history[:, 5], "y", label="3miss")
    plt.plot(history[:, 0], history[:, 4], "m", label="4miss")
    plt.plot(history[:, 0], history[:, 10], "r", label="バランス精度")
    plt.xticks(np.arange(0, num_epochs + 1, unit))
    plt.xlabel("繰り返し回数")
    plt.ylabel("精度")
    plt.title("学習曲線(精度)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{which_data}_acc.png"))
    plt.show()

