## Script for downloading data

# GloVe Vectors
wget -P data http://nlp.stanford.edu/data/glove.6B.zip
unzip data/glove.6B.zip -d data/glove
rm data/glove.6B.zip

# Questions
wget -P data http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip data/v2_Questions_Train_mscoco.zip -d data
rm data/v2_Questions_Train_mscoco.zip

wget -P data http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip data/v2_Questions_Val_mscoco.zip -d data
rm data/v2_Questions_Val_mscoco.zip

wget -P data http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip data/v2_Questions_Test_mscoco.zip -d data
rm data/v2_Questions_Test_mscoco.zip

# Annotations
wget -P data http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip data/v2_Annotations_Train_mscoco.zip -d data
rm data/v2_Annotations_Train_mscoco.zip

wget -P data http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip data/v2_Annotations_Val_mscoco.zip -d data
rm data/v2_Annotations_Val_mscoco.zip

# Image Features
wget -P data https://storage.googleapis.com/bottom-up-attention/trainval_36.zip
unzip data/trainval_36.zip -d data
rm data/trainval_36.zip

wget -P data https://storage.googleapis.com/bottom-up-attention/test2015_36.zip
unzip data/test2015_36.zip -d data
rm data/test2015_36.zip

# Image
#wget -P data http://msvocds.blob.core.windows.net/coco2015/test2015.zip
#unzip data/test2015.zip -d data
#rm data/test2015.zip
