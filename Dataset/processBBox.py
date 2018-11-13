import os
import glob
'''
To better fit the SSD input and output format,we process the SynthText data(image_name and four coordinates)
and store the process data as a .txt file,in this .txt each line is formatted as:
image_path bbox_num x1_min y1_min w1 h1 x2_min y2_min w2 h2 .........

Such as '1/ant+hill_113_90.jpg 2 365.636 418.149 439.275 438.116 448.093 414.372 509.974 438.203'
'''

rootDir = '/home/cv_whj/codes/TextBoxes-TensorFlow/dataset/SynthText/'
trainList = os.path.join(rootDir,'list_train,txt')
valList = os.path.join(rootDir,'list_val.txt')

trainList_processed = os.path.join(rootDir,'list_train_processed,txt')
valList_processed = os.path.join(rootDir,'list_val_processed,txt')

def getImageNameListPath(listPath = valList):
    '''
    :param listPath: the list of images path
    :return: (lines:the relative path of images,gt_list:the abs path of gt)
    '''
    gt_list = []
    with open(valList) as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            line = line.strip().strip('./')
            lines[i] = line
            line = os.path.join(rootDir,line)
            gt_list.append(line.split('.')[0] + '.gt')


    return lines,gt_list

def getLine(gtFileName):
    '''
    In this func, we will get the infomation of the gtFile,and return the info in following format
    numObject xmin1 ymin1 xmax1  ymax1 xmin2 ymin2 xmax2 ymax2 ......
    '''
    with open(gtFileName) as f:
        lines = f.readlines()

    outputlist = []
    numObject = int(lines[0].strip())

    outputlist.append(str(numObject))
    for i in range(numObject):
        coordinates = lines[i+1].strip().split()[1:]  #len(coordinates) = 8
        x_min = max(0.0,min(float(coordinates[0]),float(coordinates[6])))
        y_min = max(0.0,min(float(coordinates[1]),float(coordinates[3])))
        x_max = max(float(coordinates[2]),float(coordinates[4]))
        y_max = max(float(coordinates[5]), float(coordinates[7]))
        outputlist.append(str(x_min))
        outputlist.append(str(y_min))
        outputlist.append(str(x_max))
        outputlist.append(str(y_max))
    return ' '.join(outputlist)

def getProcessedFile(listPath = valList , outputTxt = valList_processed):
    lines, gt_list = getImageNameListPath(listPath)

    f = open(outputTxt, 'a+')

    for i,val in enumerate(gt_list):
        imagePath = lines[i]
        f.write(imagePath+' '+ getLine(val) + '\n')
    f.close()


#because the SynthText is to big,we can use this function to get a tiny dataset processed data
def getTinyDataset():
    imageDirName = 1
    lenRootName = len(rootDir)

    imageDir = os.path.join(rootDir,str(imageDirName))
    imageFile_list = glob.glob(os.path.join(imageDir,'*.jpg'))
    imageFile_list = sorted(imageFile_list)
    gt_list = glob.glob(os.path.join(imageDir,'*.gt'))
    gt_list = sorted(gt_list)

    train_val_ratio = 4
    image_num = len(imageFile_list)
    val_num = int(image_num/(train_val_ratio+1))

    val_image_list = imageFile_list[:val_num]
    train_image_list = imageFile_list[val_num:]
    val_gt_list = gt_list[:val_num]
    train_gt_list = gt_list[val_num:]

    outPut_train_name = 'tiny_list_train'+str(imageDirName)+'.txt'
    outPut_val_name = 'tiny_list_val' + str(imageDirName) + '.txt'

    f = open(outPut_train_name, 'a+')
    for i,val in enumerate(train_gt_list):
        imagePath = train_image_list[i][lenRootName:]
        f.write(imagePath+' '+ getLine(val) + '\n')
    f.close()

    f = open(outPut_val_name, 'a+')
    for i, val in enumerate(val_gt_list):
        imagePath = val_image_list[i][lenRootName:]
        f.write(imagePath + ' ' + getLine(val) + '\n')
    f.close()











if __name__ == '__main__':
    # getProcessedFile(listPath=trainList, outputTxt=trainList_processed)
    getTinyDataset()

