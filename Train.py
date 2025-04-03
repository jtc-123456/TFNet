import Net
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from WER import WerScore
import os
import DataProcessMoudle
import videoAugmentation
import numpy as np
import decode
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from evaluation import evaluteMode
from evaluationT import evaluteModeT
import random

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def stable(dataloader, seed):
    seed_torch(seed)
    return dataloader

def train(configParams, isTrain=True, isCalc=False):
    # 参数初始化
    # 读入数据路径
    trainDataPath = configParams["trainDataPath"]
    validDataPath = configParams["validDataPath"]
    testDataPath = configParams["testDataPath"]
    # 读入标签路径
    trainLabelPath = configParams["trainLabelPath"]
    validLabelPath = configParams["validLabelPath"]
    testLabelPath = configParams["testLabelPath"]
    # 读入模型参数
    bestModuleSavePath = configParams["bestModuleSavePath"]
    currentModuleSavePath = configParams["currentModuleSavePath"]
    # 读入参数
    device = configParams["device"]
    hiddenSize = int(configParams["hiddenSize"])
    lr = float(configParams["lr"])
    batchSize = int(configParams["batchSize"])
    numWorkers = int(configParams["numWorkers"])
    pinmMemory = bool(int(configParams["pinmMemory"]))
    moduleChoice = configParams["moduleChoice"]
    dataSetName = configParams["dataSetName"]
    max_num_states = 1

    if dataSetName == "RWTH":
        sourcefilePath = './evaluation/wer/evalute'

        if isTrain:
            fileName = "output-hypothesis-{}.ctm".format('dev')
        else:
            fileName = "output-hypothesis-{}.ctm".format('test')
        filePath = os.path.join(sourcefilePath, fileName)
    elif dataSetName == "RWTH-T":
        sourcefilePath = './evaluationT/wer/evalute'

        if isTrain:
            fileName = "output-hypothesis-{}.ctm".format('dev')
        else:
            fileName = "output-hypothesis-{}.ctm".format('test')
        filePath = os.path.join(sourcefilePath, fileName)

    # 预处理语言序列
    word2idx, wordSetNum, idx2word = DataProcessMoudle.Word2Id(trainLabelPath, validLabelPath, testLabelPath, dataSetName)
    # 图像预处理
    transform = videoAugmentation.Compose([
        videoAugmentation.RandomCrop(224),
        videoAugmentation.RandomHorizontalFlip(0.5),
        videoAugmentation.ToTensor(),
        videoAugmentation.TemporalRescale(0.2),
    ])

    transformTest = videoAugmentation.Compose([
        videoAugmentation.CenterCrop(224),
        videoAugmentation.ToTensor(),
    ])

    # 导入数据
    trainData = DataProcessMoudle.MyDataset(trainDataPath, trainLabelPath, word2idx, dataSetName, isTrain=True, transform=transform)

    validData = DataProcessMoudle.MyDataset(validDataPath, validLabelPath, word2idx, dataSetName, transform=transformTest)

    testData = DataProcessMoudle.MyDataset(testDataPath, testLabelPath, word2idx, dataSetName, transform=transformTest)

    trainLoader = DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True, num_workers=numWorkers,
                             pin_memory=pinmMemory, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)
    validLoader = DataLoader(dataset=validData, batch_size=1, shuffle=False, num_workers=numWorkers,
                             pin_memory=pinmMemory, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)
    testLoader = DataLoader(dataset=testData, batch_size=1, shuffle=False, num_workers=numWorkers,
                            pin_memory=pinmMemory, collate_fn=DataProcessMoudle.collate_fn, drop_last=True)

    # 定义模型
    moduleNet = Net.moduleNet(hiddenSize, wordSetNum * max_num_states + 1, moduleChoice, device, dataSetName, True)
    moduleNet = moduleNet.to(device)

    # 损失函数定义
    PAD_IDX = 0
    if "MSTNet" == moduleChoice:
        ctcLoss = nn.CTCLoss(blank=PAD_IDX, reduction='mean', zero_infinity=True)
    elif "VAC" == moduleChoice or "CorrNet" == moduleChoice or "MAM-FSD" == moduleChoice \
         or "SEN" == moduleChoice or "TFNet" == moduleChoice:
        ctcLoss = nn.CTCLoss(blank=PAD_IDX, reduction='none', zero_infinity=True)
        kld = DataProcessMoudle.SeqKD(T=8)
        if "MAM-FSD" == moduleChoice:
            mseLoss = nn.MSELoss(reduction="mean")

    logSoftMax = nn.LogSoftmax(dim=-1)
    # 优化函数
    params = list(moduleNet.parameters())

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0001)
    # 读取预训练模型参数
    bestLoss = 65535
    bestLossEpoch = 0
    bestWerScore = 65535
    bestWerScoreEpoch = 0
    epoch = 0

    lastEpoch = -1
    if os.path.exists(currentModuleSavePath):
        checkpoint = torch.load(currentModuleSavePath, map_location=torch.device('cpu'), weights_only=False)
        moduleNet.load_state_dict(checkpoint['moduleNet_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        bestLoss = checkpoint['bestLoss']
        bestLossEpoch = checkpoint['bestLossEpoch']
        bestWerScore = checkpoint['bestWerScore']
        bestWerScoreEpoch = checkpoint['bestWerScoreEpoch']
        epoch = checkpoint['epoch']
        lastEpoch = epoch
        print(
            f"已加载预训练模型 epoch: {epoch}, bestLoss: {bestLoss:.5f}, bestEpoch: {bestLossEpoch}, werScore: {bestWerScore:.5f}, bestEpoch: {bestWerScoreEpoch}")
    else:
        print(
            f"未加载预训练模型 epoch: {epoch}, bestLoss: {bestLoss}, bestEpoch: {bestLossEpoch}, werScore: {bestWerScore:.5f}, bestEpoch: {bestWerScoreEpoch}")

    # 设置学习率衰减规则
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=[35, 45],
                                                     gamma=0.2, last_epoch=lastEpoch)

    # 解码参数
    decoder = decode.Decode(word2idx, wordSetNum + 1, 'beam')

    if isTrain:
        print("开始训练模型")
        # 训练模型
        epochNum = 55

        if -1 != lastEpoch:
            epochN = epochNum - lastEpoch
        else:
            epochN = epochNum

        seed = 1
        for _ in range(epochN):
            moduleNet.train()

            scaler = GradScaler()
            loss_value = []
            optimizer.zero_grad()
            for Dict in tqdm(stable(trainLoader, seed + epoch)):
                data = Dict["video"].to(device)
                label = Dict["label"]
                dataLen = Dict["videoLength"]
                ##########################################################################
                targetOutData = [torch.tensor(yi).to(device) for yi in label]
                targetLengths = torch.tensor(list(map(len, targetOutData)))
                targetOutData = torch.cat(targetOutData, dim=0).to(device)

                with autocast():
                    logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, x1, x2, x3 = moduleNet(data, dataLen, True)
                    #########################################
                    if "MSTNet" == moduleChoice:
                        logProbs1 = logSoftMax(logProbs1)
                        logProbs2 = logSoftMax(logProbs2)
                        logProbs3 = logSoftMax(logProbs3)
                        logProbs4 = logSoftMax(logProbs4)

                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths)
                        loss3 = ctcLoss(logProbs3, targetOutData, lgt * 2, targetLengths)
                        loss4 = ctcLoss(logProbs4, targetOutData, lgt * 4, targetLengths)
                        loss = loss1 + loss2 + loss3 + loss4
                    elif "VAC" == moduleChoice or "CorrNet" == moduleChoice or "MAM-FSD" == moduleChoice \
                            or "SEN" == moduleChoice or "TFNet" == moduleChoice:
                        loss3 = 25 * kld(logProbs2, logProbs1, use_blank=False)

                        logProbs1 = logSoftMax(logProbs1)
                        logProbs2 = logSoftMax(logProbs2)

                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths).mean()
                        loss2 = ctcLoss(logProbs2, targetOutData, lgt, targetLengths).mean()
                        if "MAM-FSD" == moduleChoice:
                            loss4 = mseLoss(x1[0], x1[1])
                            loss5 = mseLoss(x2[0], x2[1])
                            loss6 = mseLoss(x3[0], x3[1])

                            loss = loss1 + loss2 + loss3 + 5 * loss4 + 1 * loss5 + 70 * loss6
                        elif "TFNet" == moduleChoice:
                            loss6 = 25 * kld(logProbs4, logProbs3, use_blank=False)

                            logProbs3 = logSoftMax(logProbs3)
                            logProbs4 = logSoftMax(logProbs4)

                            loss4 = ctcLoss(logProbs3, targetOutData, lgt, targetLengths).mean()
                            loss5 = ctcLoss(logProbs4, targetOutData, lgt, targetLengths).mean()

                            logProbs5 = logSoftMax(logProbs5)
                            loss7 = ctcLoss(logProbs5, targetOutData, lgt, targetLengths).mean()

                            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
                        else:
                            loss = loss1 + loss2 + loss3

                    if np.isinf(loss.item()) or np.isnan(loss.item()):
                        print('loss is nan')
                        continue

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                loss_value.append(loss.item())

                torch.cuda.empty_cache()

            print("epoch: %d, trainLoss: %.5f, lr: %f" % (
            epoch, np.mean(loss_value), optimizer.param_groups[0]['lr']))

            epoch = epoch + 1

            scheduler.step()

            moduleNet.eval()
            print("开始验证模型")
            # 验证模型
            werScoreSum = 0
            total_info = []
            total_sent = []
            loss_value = []
            for Dict in tqdm(validLoader):
                data = Dict["video"].to(device)
                label = Dict["label"]
                dataLen = Dict["videoLength"]
                info = Dict["info"]
                ##########################################################################
                targetOutData = [torch.tensor(yi).to(device) for yi in label]
                targetLengths = torch.tensor(list(map(len, targetOutData)))
                targetData = targetOutData
                targetOutData = torch.cat(targetOutData, dim=0).to(device)
                batchSize = len(targetLengths)

                with torch.no_grad():
                    logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, x1, x2, x3 = moduleNet(data, dataLen, False)

                    logProbs1 = logSoftMax(logProbs1)

                    if "MSTNet" == moduleChoice:
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths)
                    else:
                        loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths).mean()

                    loss = loss1

                    if np.isinf(loss.item()) or np.isnan(loss.item()):
                        print('loss is nan')
                        continue

                loss_value.append(loss.item())
                ##########################################################################
                pred, targetOutDataCTC = decoder.decode(logProbs1, lgt, batch_first=False, probs=False)

                if dataSetName == "RWTH" or dataSetName == "RWTH-T":
                    total_info += info
                    total_sent += pred
                elif dataSetName == "CSL-Daily" or dataSetName == "CE-CSL":
                    werScore = WerScore([targetOutDataCTC], targetData, idx2word, batchSize)
                    werScoreSum = werScoreSum + werScore

            torch.cuda.empty_cache()

            currentLoss = np.mean(loss_value)

            werScore = werScoreSum / len(validLoader)

            if werScore < bestWerScore:
                bestWerScore = werScore
                bestWerScoreEpoch = epoch - 1

                moduleDict = {}
                moduleDict['moduleNet_state_dict'] = moduleNet.state_dict()
                moduleDict['optimizer_state_dict'] = optimizer.state_dict()
                moduleDict['bestLoss'] = bestLoss
                moduleDict['bestLossEpoch'] = bestLossEpoch
                moduleDict['bestWerScore'] = bestWerScore
                moduleDict['bestWerScoreEpoch'] = bestWerScoreEpoch
                moduleDict['epoch'] = epoch
                torch.save(moduleDict, bestModuleSavePath)

            bestLoss = currentLoss
            bestLossEpoch = epoch - 1

            moduleDict = {}
            moduleDict['moduleNet_state_dict'] = moduleNet.state_dict()
            moduleDict['optimizer_state_dict'] = optimizer.state_dict()
            moduleDict['bestLoss'] = bestLoss
            moduleDict['bestLossEpoch'] = bestLossEpoch
            moduleDict['bestWerScore'] = bestWerScore
            moduleDict['bestWerScoreEpoch'] = bestWerScoreEpoch
            moduleDict['epoch'] = epoch
            torch.save(moduleDict, currentModuleSavePath)

            moduleSavePath1 = 'module/bestMoudleNet_' + str(epoch) + '.pth'
            torch.save(moduleDict, moduleSavePath1)

            if dataSetName == "RWTH":
                ##########################################################################
                DataProcessMoudle.write2file(filePath, total_info, total_sent)

                evaluteMode('evalute_dev1')
                ##########################################################################
                DataProcessMoudle.write2file('./wer/' + "output-hypothesis-{}{:0>4d}.ctm".format('dev', epoch), total_info, total_sent)
            elif dataSetName == "RWTH-T":
                ##########################################################################
                DataProcessMoudle.write2file(filePath, total_info, total_sent)
                evaluteModeT('evalute_dev1')
                ##########################################################################
                DataProcessMoudle.write2file('./wer/' + "output-hypothesis-{}{:0>4d}.ctm".format('dev', epoch),
                                             total_info, total_sent)

            print(f"validLoss: {currentLoss:.5f}, werScore: {werScore:.2f}")
            print(f"bestLoss: {bestLoss:.5f}, beatEpoch: {bestLossEpoch}, bestWerScore: {bestWerScore:.2f}, bestWerScoreEpoch: {bestWerScoreEpoch}")
    else:
        bestWerScore = 65535
        offset = 1
        for i in range(1):
            currentModuleSavePath = "module/bestMoudleNet_" + str(i + offset) + ".pth"
            checkpoint = torch.load(currentModuleSavePath, map_location=torch.device('cpu'),weights_only=False)
            moduleNet.load_state_dict(checkpoint['moduleNet_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            moduleNet.eval()
            print("开始验证模型")
            # 验证模型
            werScoreSum = 0
            loss_value = []
            total_info = []
            total_sent = []
            total_target=[]

            for Dict in tqdm(testLoader):
                data = Dict["video"].to(device)
                label = Dict["label"]
                dataLen = Dict["videoLength"]
                info = Dict["info"]
                ##########################################################################
                targetOutData = [torch.tensor(yi).to(device) for yi in label]
                targetLengths = torch.tensor(list(map(len, targetOutData)))
                targetData = targetOutData
                targetOutData = torch.cat(targetOutData, dim=0).to(device)
                batchSize = len(targetLengths)

                with torch.no_grad():
                    logProbs1, logProbs2, logProbs3, logProbs4, logProbs5, lgt, x1, x2, x3 = moduleNet(data, dataLen, False)

                    logProbs1 = logSoftMax(logProbs1)

                    loss1 = ctcLoss(logProbs1, targetOutData, lgt, targetLengths).mean()

                    loss = loss1

                loss_value.append(loss.item())

                pred, targetOutDataCTC = decoder.decode(logProbs1, lgt, batch_first=False, probs=False)

                if dataSetName == "RWTH" or dataSetName == "RWTH-T":
                    total_info += info
                    total_sent += pred
                elif dataSetName == "CSL-Daily" or dataSetName == "CE-CSL":
                    total_info += info
                    total_sent += pred
                    total_target+= label                 
                    werScore = WerScore([targetOutDataCTC], targetData, idx2word, batchSize)
                    werScoreSum = werScoreSum + werScore

                torch.cuda.empty_cache()

            currentLoss = np.mean(loss_value)

            werScore = werScoreSum / len(testLoader)

            if werScore < bestWerScore:
                bestWerScore = werScore
                bestWerScoreEpoch = i + offset - 1

            bestLoss = currentLoss
            bestLossEpoch = i + offset - 1

            if dataSetName == "RWTH":
                ##########################################################################
                DataProcessMoudle.write2file(filePath, total_info, total_sent)
                evaluteMode('evalute_test')
                ##########################################################################
                DataProcessMoudle.write2file('./wer/' + "output-hypothesis-{}{:0>4d}.ctm".format('test', i+1), total_info,
                                             total_sent)
            elif dataSetName == "RWTH-T":
                ##########################################################################
                DataProcessMoudle.write2file(filePath, total_info, total_sent)
                evaluteModeT('evalute_test')
                ##########################################################################
                DataProcessMoudle.write2file('./wer/' + "output-hypothesis-{}{:0>4d}.ctm".format('test', i+1), total_info,
                                             total_sent)
            if dataSetName == "CE-CSL":
              import pandas as pd

              # # 把预测和真实标签转成字符串（用空格连接）
              pred_strs = ["/".join([word[0] for word in seq] ) for seq in total_sent]
              target_strs = ["/".join([idx2word[idx.item()] for idx in tgt]) for tgt in total_target]


              # 保存为 CSV
              df = pd.DataFrame({
                  "video": total_info,
                  "predict": pred_strs,
                  "target":target_strs
              })
              save_csv_path = f"./predict_vs_target_{dataSetName.lower()}.csv"
              df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
              print(f"🔄 保存预测结果到: {save_csv_path}")
            print(f"testLoss: {currentLoss:.5f}, werScore: {werScore:.2f}")
            print(f"bestLoss: {bestLoss:.5f}, bestEpoch: {bestLossEpoch}, bestWerScore: {bestWerScore:.2f}, bestWerScoreEpoch: {bestWerScoreEpoch}")


