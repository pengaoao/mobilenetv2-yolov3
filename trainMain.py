"""
Img with no box label, how?
random img flip, target should also be flipped

Notes:
need to use leaky relu; no relu before yolo layer
"""

import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import argparse
import dataset
import model_sum as model

import matplotlib.pyplot as plt
import torch
import imgUtils
import utils

MODE="predict"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDatasetListFile', type=str, default="train.txt",
                        help='training dataset list file')
    parser.add_argument('--trainDatasetDirectory', type=str, default="./data/images",
                        help='training dataset directory')
    parser.add_argument('--trainDatasetLabelDirectory', type=str, default="./data/labels",
                        help='training dataset directory')
    
    parser.add_argument('--imgSquareSize', type=int, default=416,
                        help='Padded squared image size length')
    parser.add_argument('--batchSize', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--pretrainedParamFile', type=str, default="yoloParam390.dict",
                        help='Pretrained parameter file')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    options=parse_args()
    
    trainDataSet=dataset.ListDataset(options)
    device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
    net=model.objDetNet(options)
    net.to(device)
    net.loadPretrainedParams()
    
    if MODE is "train":
        dataloaderTrain = DataLoader(
                trainDataSet,
                batch_size=options.batchSize,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                collate_fn=trainDataSet.collate_fn
            )
        optimizer=optim.Adam(net.parameters(),lr=0.0005,eps=1e-3)
        _trainCount=0
        Epoch=5
        for _ in range(Epoch):
            print("New Epoch:",_)
            temp = 0
            for inputs,labels in dataloaderTrain:
               # print("inputs",inputs)
                optimizer.zero_grad()
                inputs=Variable(inputs.to(device))
                labels=(Variable(torch.cat(labels,dim=0).to(device)) if labels!=[] else 
                        Variable(torch.FloatTensor(0,6).to(device)))
                out,loss=net(inputs,labels)
                # if _trainCount%5==0:
                print(_trainCount, loss.item())

                loss.backward()
                optimizer.step()
                _trainCount+=1
                temp+=1
                if _trainCount%10==0 and _trainCount >100:
                        torch.save(net.state_dict(),"yoloParam%d.dict"%_trainCount)
            print("temp",temp)
    elif MODE is "predict":
        fileName='./data/images/BloodImage_00000.jpg'
        net.eval()
        img=Variable(trainDataSet.imgRead(fileName).unsqueeze(0).to(device))
        with torch.no_grad():
            out,_=net(img)
        pred=torch.cat(out,dim=1).cpu()
        print(pred.shape)
        detections = utils.non_max_suppression(pred, 0.4, 0.2)[0]
        if detections is None:
            print("can not find the red cell")
            exit()
        a,label=torch.split(detections,[6,1],dim=1)
        label=torch.cat([torch.zeros(label.shape[0],1),label,a],dim=1)
        label[:,2:6]=utils.xyxy2xywh(label[:,2:6])/options.imgSquareSize

        imgUtils.showImgNLab(img[0],label)
        
    
    
                    
                    
            
        
    