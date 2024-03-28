#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pickle
#from ipysheet import from_array, to_array, sheet as isheet
import matplotlib.pyplot as plt
import torch
import numpy as np
#导入内置os模块
import os

#创建单层目录
def mkdir_single(path):
    #目录名称
    #basename:返回目录路径中的最后一个元素
    dirName = os.path.basename(path)
    # 判断路径是否存在
    isExists=os.path.exists(path)

    if not isExists:
        # 如果不存在，则创建单层目录
        os.mkdir(path) 
        print('目录创建成功：' + dirName )
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print('目录已存在：' + dirName )
        return False
def mkdir_multi(path):
    # 判断路径是否存在
    isExists=os.path.exists(path)

    if not isExists:
        # 如果不存在，则创建目录（多层）
        os.makedirs(path) 
        print('目录创建成功！')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print('目录已存在！')
        return False


# import pickle
def save_variable(v,filename):
    f=open(filename,'wb')
    print(filename)
    pickle.dump(v,f)
    f.close()
    return filename
 
def load_variable(filename):
    f=open(filename,'rb')
    print(filename)
    r=pickle.load(f)
    f.close()
    return r
"""# if __name__=='__main__':
#     c = [1, 2, 3, 4, 5, 6, 7]
#     filename=save_variable(c,'D:\\test.txt')
#     d=load_variavle(filename)
#     print(d==c)"""


#from ipysheet import from_array, to_array, sheet as isheet
def display_data(data):
    sheet = from_array(data.cpu().numpy())
    sheet.cells[0].numeric_format = '0.0000'
    sheet.layout.height = '200px'
    sheet.layout.width = '1500px'
    return sheet


# import matplotlib.pyplot as plt
# import numpy as np
def show_corr_level(D1, D2, D3):
    corr_D1 = np.abs(np.corrcoef(D1.T.cpu()))
    corr_D2 = np.abs(np.corrcoef(D2.T.cpu()))
    corr_D3 = np.abs(np.corrcoef(D3.T.cpu()))
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,3,1)
    plt.imshow(corr_D1, vmin=0, vmax=1)
    ax = fig.add_subplot(1,3,2)
    plt.imshow(corr_D2, vmin=0, vmax=1)
    ax = fig.add_subplot(1,3,3)
    plt.imshow(corr_D3, vmin=0, vmax=1)
    
    
# import torch
def pinv_eq(A):
    m, n = A.shape
    if m >= n:
        A_pinv = torch.mm(torch.inverse(torch.mm(A.T, A)), A.T)
    if m < n:
        A_pinv = torch.mm(A.T, torch.inverse(torch.mm(A, A.T)))
    return A_pinv

def eval_results(predict, labels,class_num):
    predict=np.array(predict)
    labels=np.array(labels)
    from sklearn.metrics import cohen_kappa_score
    kappa = np.zeros(1)
    kappa= cohen_kappa_score(predict, labels)
    
    correct =(predict == labels).sum().item()
    OA=correct/len(labels)
    
    class_correct = list(0. for i in range(class_num))
    class_total = list(0. for i in range(class_num))
    acc = list(0. for i in range(class_num))
    c = (predict == labels).squeeze()
    for i in range(len(labels)):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
    for i in range(class_num):
        acc[i]=class_correct[i] / class_total[i]
        AA=np.mean(acc)
        
    return OA, AA, kappa, acc

def eval_results_own(predict, labels,class_num):
    predict=np.array(predict)
    labels=np.array(labels)
    
    #confusion matrix
    cm=np.zeros([class_num,class_num])
    for i in range(class_num):
        for j in range(class_num):
            for n in range(len(predict)):
                if(predict[n]==i and labels[n]==j):
                    cm[i,j]+=1
    cm.astype(int)    
    
    oa=np.diag(cm).sum()/cm.sum()
    
    acc=np.zeros([class_num])
    for i in range(class_num):
        acc[i]=cm[i,i]/cm[:,i].sum()
    aa=np.mean(acc)
    
    pe=(np.sum(cm,0)*np.sum(cm,1)).sum()/cm.sum()**2
    kappa=(oa-pe)/(1-pe)
        
    return oa, aa, kappa, acc,cm

def spatial_filter(testmap,groundtruth,window):
    [h,w]=groundtruth.shape
    newmap=testmap.copy()
#     print(newmap.shape)
    for i in range(h):
#         print(i)
        hd= i-window
        if hd<0:
            hd=0 
        hu= i+window+1
        if hu>h:
            hu=h
        
        for j in range(w):
            wd = j -window
            if wd < 0:
                wd = 0    
            wu = j +window+1
            if wu>w:
                wu=w
#             print(j)
            tspatch= newmap[hd:hu,wd:wu]
            tsl=tspatch[np.where(tspatch)]
        
            if (len(tsl)!=0) and (groundtruth[i,j]!=0):
                newmap[i,j] = np.argmax(np.bincount(tsl))
    return newmap

def spatial_filter_different(testmap,groundtruth,window1,window2,dif=[]):
    [h,w]=groundtruth.shape
    newmap=testmap.copy()
#     newmap=np.zeros([h,w])
#     print(newmap.shape)
    for i in range(h):
#         print(i)
        
        for j in range(w):
            if groundtruth[i,j] in dif:
                window=window2
            else:
                window=window1
                
            hd= i-window
            if hd<0:
                hd=0 
            hu= i+window+1
            if hu>h:
                hu=h
        
            wd = j -window
            if wd < 0:
                wd = 0    
            wu = j +window+1
            if wu>w:
                wu=w
#             print(j)
            tspatch= newmap[hd:hu,wd:wu]
#             tspatch= testmap[hd:hu,wd:wu]
            tsl=tspatch[np.where(tspatch)]
        
            if (len(tsl)!=0) and (groundtruth[i,j]!=0):
                newmap[i,j] = np.argmax(np.bincount(tsl))
    return newmap