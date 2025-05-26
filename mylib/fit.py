import os

import torch
import numpy as np
from tqdm import tqdm

from .make_confusion_matrix import make_confusion_matrix


def fit(
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
):
    base_epochs = len(history)
    # print("base_epochs =", base_epochs)
    

    for epoch in range(base_epochs, num_epochs + base_epochs):
        # print("epoch =", epoch)

        n_train = 0 # trainで扱った画像の数
        n_train_acc = 0 # 扱った画像のうち予測が当たっていた数
        n_train_acc2 = 0 # 2番目までの予測を許容した場合の数
        n_train_acc3 = 0 # 3番目までの予測を許容した場合の数
        train_loss = 0 # trainでの損失関数


        n_test = 0 # testで扱った画像の数
        n_test_acc = 0 # 扱った画像のうち予測が当たっていた数
        n_test_acc2 = 0 # 2番目までの予測を許容した場合の数
        n_test_acc3 = 0 # 3番目までの予測を許容した場合の数
        val_loss = 0 # testでの損失関数

        ## 4ビット中0〜4ビットが一致しているサンプルの数をそれぞれn_val_acc[0]からn_val_acc[4]に追加
        n_val_acc = np.array([0, 0, 0, 0, 0])
        

        # balanced_accuracyを計算するための辞書
        # 各キーの役割:
        # label_count_train: 各ラベルの登場回数 (訓練データ)
        # label_correct_train: 各ラベルで正解した回数 (訓練データ)
        # label_count_test: 各ラベルの登場回数 (テストデータ)
        # label_correct_test: 各ラベルで正解した回数 (テストデータ)
        # train_BA: 訓練データでのバランス精度
        # test_BA: テストデータでのバランス精度

        balanced_acc_dict = {
            "label_count_train": {str(i1): 0 for i1 in range(0, len_classes)},
            "label_correct_train": {str(i1): 0 for i1 in range(0, len_classes)},
            "label_count_test": {str(i1): 0 for i1 in range(0, len_classes)},
            "label_correct_test": {str(i1): 0 for i1 in range(0, len_classes)},
            "train_BA": 0,
            "test_BA": 0,
        }
        # print("\033[91mbalanced_acc_dict =\033[0m\n", balanced_acc_dict)
        # print("balanced_acc_dict[\"label_count_train\"] =", balanced_acc_dict["label_count_train"])
        # print("balanced_acc_dict[\"label_correct_train\"] =", balanced_acc_dict["label_correct_train"])
        # print("balanced_acc_dict[\"label_count_test\"] =", balanced_acc_dict["label_count_test"])
        # print("balanced_acc_dict[\"label_correct_test\"] =", balanced_acc_dict["label_correct_test"])
        # print("balanced_acc_dict[\"train_BA\"] =", balanced_acc_dict["train_BA"])
        # print("balanced_acc_dict[\"test_BA\"] =", balanced_acc_dict["test_BA"])

        net.train()
        
        #disable=Trueではプログレスバーを表示しない
        for inputs_train, labels_train in tqdm(train_loader, disable=True):
            train_batch_size = len(labels_train)  # = batch_size in main.py
            n_train += train_batch_size
            # print("n_train =", n_train)

            ## train_loaderのデータはGPUに入れないといけない
            inputs_train = inputs_train.to(device)
            labels_train = labels_train.to(device)
            # print("inputs_train.shape =", inputs_train.shape)
            # print("\033[91minputs_train =\033[0m\n", inputs_train)
            # print("labels_train =", labels_train)
            # print("labels_train.shape =", labels_train.shape) # [batch_size]
            optimizer.zero_grad()
             
            outputs = net(inputs_train)

            # print("outputs.shape =", outputs.shape) # [batch_size, len_classes]
            # print("outputs[0] =", outputs[0])
            # e.g. outputs[0] = tensor([-0.0607, -0.1258,  0.6659, -0.2093, -0.2549,  0.0196,  0.2348,  0.0521, -0.3396], 
            #                           device='cuda:1', grad_fn=<SelectBackward0>)
            # この場合 0番目のデータラベル2と予測される

            loss_train = criterion(outputs, labels_train)
            # print("loss_train =", loss_train)

            loss_train.backward()
            # print("loss_train.items() =", loss_train.item())
            
            optimizer.step()
            

            # 各indexを確率順に並び替えた後に、確率値とラベルをまとめる ## 今は使ってない
            predicted_train_top_len_classes = torch.topk(outputs, len_classes, dim=1) #[0]が確率値、[1]がラベル
            # print("\033[91mpredicted_train_top_len_classes =\033[0m\n", predicted_train_top_len_classes)

            topk = min(3, len_classes) #基本3、クラス数が2個の場合だけ2
            # 各indexのtop3の確率値とラベルをまとめる
            predicted_train_top3 = torch.topk(outputs, topk, dim=1) #[0]が確率値、[1]がラベル shape: [train_batch_size, topk]

            # 各indexの最大、2番目、3番目の予測の確率値(softmax前、負の値もある)をまとめる
            predicted_train_1st_values = predicted_train_top3[0][:, 0] # shape: [train_batch_size]
            predicted_train_2nd_values = predicted_train_top3[0][:, 1]
            predicted_train_3rd_values = predicted_train_top3[0][:, topk-1]
            # print("predicted_train_1st_values =", predicted_train_1st_values)
            # print("predicted_train_2nd_values =", predicted_train_2nd_values)
            # print("predicted_train_3rd_values =", predicted_train_3rd_values)

            # 各indexの最大、2番目、3番目の予測のラベルをまとめる
            predicted_train_1st_labels = predicted_train_top3[1][:, 0]
            predicted_train_2nd_labels = predicted_train_top3[1][:, 1]
            predicted_train_3rd_labels = predicted_train_top3[1][:, topk-1]
            # print("predicted_train_1st_labels =", predicted_train_1st_labels)
            # print("predicted_train_2nd_labels =", predicted_train_2nd_labels)
            # print("predicted_train_3rd_labels =", predicted_train_3rd_labels)
            
            # lossをtrain_batch_sizeで割った平均計算が行われているので平均前の損失に戻して加算
            train_loss += loss_train.item() * train_batch_size
            n_train_acc += (predicted_train_1st_labels == labels_train).sum().item()
            n_train_acc2 += ((predicted_train_1st_labels == labels_train) | 
                             (predicted_train_2nd_labels == labels_train)).sum().item()
            n_train_acc3 += ((predicted_train_1st_labels == labels_train) | 
                             (predicted_train_2nd_labels == labels_train) | 
                             (predicted_train_3rd_labels == labels_train)).sum().item()

            # print("n_train_acc =", n_train_acc)
            # print("n_train_acc2 =", n_train_acc2)
            # print("n_train_acc3 =", n_train_acc3)
            
            #バランス精度計算
            for i in range(train_batch_size):
                # print(f"labels_train[{i}] =", labels_train[i])
                balanced_acc_dict["label_count_train"][str(labels_train[i].item())] += 1
                if predicted_train_1st_labels[i].item() == labels_train[i].item():
                    balanced_acc_dict["label_correct_train"][str(predicted_train_1st_labels[i].item())] += 1
            # print("\033[91mbalanced_acc_dict =\033[0m\n", balanced_acc_dict)
            # print("balanced_acc_dict[\"label_count_train\"] =", balanced_acc_dict["label_count_train"])
            # print("balanced_acc_dict[\"label_correct_train\"] =", balanced_acc_dict["label_correct_train"])
            # print("balanced_acc_dict[\"label_count_test\"] =", balanced_acc_dict["label_count_test"])
            # print("balanced_acc_dict[\"label_correct_test\"] =", balanced_acc_dict["label_correct_test"])
            # print("balanced_acc_dict[\"train_BA\"] =", balanced_acc_dict["train_BA"])
            # print("balanced_acc_dict[\"test_BA\"] =", balanced_acc_dict["test_BA"])


        # print("n_train =", n_train)
        # print("n_train_acc =", n_train_acc)
        # print("n_train_acc2 =", n_train_acc2)
        # print("n_train_acc3 =", n_train_acc3)
        

        net.eval()

        #disable=Trueではプログレスバーを表示しない
        for inputs_test, labels_test in tqdm(test_loader, disable=True):
            test_batch_size = len(labels_test)
            n_test += test_batch_size
            # print("n_test =", n_test)

            # GPUへ転送
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            outputs_test = net(inputs_test)

            loss_test = criterion(outputs_test, labels_test)

            topk = min(3, len_classes)
            #=== ここからpredicted_test_3rd_labelsまで現状使ってない変数 ===#
            predicted_test_top3 = torch.topk(outputs_test, topk, dim=1) #[0]が確率値、[1]がラベル
            
            predicted_test_1st_values = predicted_test_top3[0][:, 0]
            predicted_test_2nd_values = predicted_test_top3[0][:, 1]
            predicted_test_3rd_values = predicted_test_top3[0][:, topk-1]

            predicted_test_1st_labels = predicted_test_top3[1][:, 0]
            predicted_test_2nd_labels = predicted_test_top3[1][:, 1]
            predicted_test_3rd_labels = predicted_test_top3[1][:, topk-1]

            # lossをtest_batch_sizeで割った平均計算が行われているので平均前の損失に戻して加算
            val_loss += loss_test.item() * test_batch_size


            n_test_acc += (predicted_test_1st_labels == labels_test).sum().item()
            n_test_acc2 += ((predicted_test_1st_labels == labels_test) | 
                             (predicted_test_2nd_labels == labels_test)).sum().item()
            n_test_acc3 += ((predicted_test_1st_labels == labels_test) | 
                             (predicted_test_2nd_labels == labels_test) | 
                             (predicted_test_3rd_labels == labels_test)).sum().item()

            ## 4ビット中0〜4ビットが一致しているサンプルの数をそれぞれn_val_acc[0]からn_val_acc[4]に追加
            for i in range(test_batch_size):
                pred_label = classes[predicted_test_1st_labels[i]]
                true_label = classes[labels_test[i]]

                # 各ビットが一致しているかを確認して、その一致数をカウント
                correct = sum(p == t for p, t in zip(pred_label, true_label))

                n_val_acc[correct] += 1

            # print("n_val_acc", n_val_acc)

            for i in range(test_batch_size):
                balanced_acc_dict["label_count_test"][str(labels_test[i].item())] += 1
                if predicted_test_1st_labels[i].item() == labels_test[i].item():
                    balanced_acc_dict["label_correct_test"][str(predicted_test_1st_labels[i].item())] += 1


        ## 精度計算
        train_acc = n_train_acc / n_train
        # 4ビット中0〜4ビットが一致しているサンプルの数をそれぞれn_val_acc[0]からn_val_acc[4]に追加
        val_acc: np.ndarray[5] = n_val_acc / n_test
        # print("val_acc =", val_acc)

        ###################
        ## test_acc = n_test_acc / n_test
        ## val_acc[4]がtest_accと同じ

        ## 損失計算
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test

        ## バランス精度の計算
        # クラスごとにrecallを取りクラス数で割り平均をとる
        for i in range(len_classes):
            balanced_acc_dict["train_BA"] += (
                balanced_acc_dict["label_correct_train"][str(i)]
                / balanced_acc_dict["label_count_train"][str(i)]
            )
            balanced_acc_dict["test_BA"] += (
                balanced_acc_dict["label_correct_test"][str(i)]
                / balanced_acc_dict["label_count_test"][str(i)]
            )
        balanced_acc_dict["train_BA"] /= len_classes
        balanced_acc_dict["test_BA"] /= len_classes

        ## 結果の表示
        print(
            f"Epoch [{(epoch+1)}/{num_epochs+base_epochs}]:\n"
            f"train_loss {avg_train_loss:.5f}, train_acc: {train_acc:.5f}, BA: {balanced_acc_dict['train_BA']:.5f} \n"
            f"val_loss: {avg_val_loss:.5f}, val_acc: {val_acc[4]:.5f}, val_BA: {balanced_acc_dict['test_BA']:.5f}"
        )

        ## 記録(少数第5位まで)
        item = np.array(
            [
                int(epoch + 1),  # `epoch` を整数型に変換
                round(avg_train_loss, 5),
                round(train_acc, 5),
                round(avg_val_loss, 5),
                *[round(acc, 5) for acc in val_acc],  # 各 val_acc の要素を少数第5位まで丸める
                round(balanced_acc_dict["train_BA"], 5),
                round(balanced_acc_dict["test_BA"], 5),
            ],
            dtype=object # 配列全体が float 型に統一されないようにする
        )

        history = np.vstack((history, item))


    # 学習後のモデルの保存
    if save_model is True:
        torch.save(
            net,
            os.path.join(
                save_dir,
                f"epoch{num_epochs}.pth"
            )
        )
    ## 混同行列の保存 ## 
    if save_cm_ls is True:
        make_confusion_matrix(
            device=device,
            epoch=num_epochs,
            classes=classes,
            test_loader=test_loader,
            save_dir=save_dir,
            net=net
        )
    return history

     