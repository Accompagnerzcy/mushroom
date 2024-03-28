#DTYPE='float32'
DTYPE='float64'
CLASS_NUM=3
# class_name=[1,5]
PATCH_SIZE=7
# num_pca=30
# whiten=True


train_data_path='../data2/train_data/3d_data'
train_label_path='../data2/train_data/label'

train_num=1000
val_num=500
test_no=5 #the no of data ,for different data processing
train_no=2
train_data_name='./sample/TrainSet_6'+'_'+str(train_num)
val_data_name='./sample/ValSet_6'+'_'+str(val_num)

test_data_path='../data2/test_data/3d_data'
test_label_path='../data2/test_data/label'

EPOCH=20 #100
BATCH_SIZE=64
LR=5e-4
model_path= '.\model'
# checkpoint_dir='.\model\checkpoint'
best_model_dir='.\model\model_best'
model_dir='.\model'
train_result_dir='.\ViT_train_result'
writer_name='runs'

band_patches=7

band=204
num_class=2
mode='ViT' 
weight_decay=0
gamma=0.9