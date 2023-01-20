import torch
from torchvision import transforms
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from pathlib import Path
from PIL import Image

def plot_confusion_matrix(conf_mat, 
                          class_names,
                          interactive = False,
                          path = "",
                          title=""):
    """
      Plot the confusion matrix
    
      Args:
        conf_mat: confusion matrix
        class_names: class names
        interactive: flag to show interactive the plot
        path: where to save the plot. It will not be saved, if kept empty 
    """
    
    sns.set()
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    #annot=True to annotate cells
    sns.heatmap(conf_mat, annot=True, ax = ax, cmap="YlGnBu", fmt='g',
                xticklabels= class_names, yticklabels= class_names)
    
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix '+title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

#    # try to put the class names on the axes
#    # this seems to work when there "not so many" classes only...
#    try :
#        ax.xaxis.set_ticklabels(class_names)
#        ax.yaxis.set_ticklabels(class_names)
#    except:
#        print(cls.bcolors.WARNING
#              + "plot_functions::plot_confusion_matrix:: Warning: error initialising the plot axes"
#              + cls.bcolors.ENDC)
        
    # show the plot, if required
    if interactive :
        plt.show()

    # save the plot, if required
    if path != "" :

        # build the complete path where to save the plot
        path_full = os.path.join(path, title+"confusion_matrix.png")
        
        # save the plot
        plt.savefig(path_full)

    # close the current plot
    plt.close()

    return


def predict(model, test_image):
    idx_to_class={0: 'browser', 1: 'calcolatrice', 2: 'firefox', 3: 'fotocamera', 4: 'galleria', 5: 'gmail', 6: 'googletalk', 7: 'maps', 8: 'market', 9: 'musica'}
    transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    test_image_tensor = transform(test_image)
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(1, dim=1)
        label = idx_to_class[topclass.cpu().numpy()[0][0]]
        return label
    
matplotlib.use('tkagg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
report=""
matmat= {
    "train":np.zeros((10,10)),
    "test":np.zeros((10,10)),
    "valid":np.zeros((10,10))
}

ltoi= {
    "browser":0,
    "calcolatrice":1,
    "firefox":2,
    "fotocamera":3,
    "galleria":4,
    "gmail":5,
    "googletalk":6,
    "maps":7,
    "market":8,
    "musica":9
}

customModel=torch.load("C:\\Users\\nicco\\Documents\\PlatformIO\Projects\\ScreenAI\\imageClassification\\tailoringOutput\\_model_15.pt")
customModel = customModel.to(device)

# Freeze model parameters
for param in customModel.parameters():
    param.requires_grad = False

dataset = os.path.join(Path(__file__).parents[0], 'iconDataset')



dirs = ['train','valid', 'test']
labels = ["browser", "calcolatrice","firefox","fotocamera","galleria","gmail","googletalk","maps","market","musica"]
for bigdir in dirs:
    bigdirPath = os.path.join(os.path.join(dataset,bigdir))
    for label in labels:
        smalldir = os.path.join(bigdirPath,label)
        images = os.listdir(smalldir)
        for image in images:
            print("teasting an image\n")
            img_path=os.path.join(smalldir,image)
            prediction=predict(customModel,Image.open(img_path))
            if prediction!=label:
                report=report+"{0} should be {1} but was identified as {2}\n".format(image,label,prediction)
            matmat[bigdir][ltoi[prediction]][ltoi[label]]=matmat[bigdir][ltoi[prediction]][ltoi[label]]+1
    plot_confusion_matrix(matmat[bigdir],labels,path="C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\\ScreenAI\\imageClassification\\tailoringOutput",title=bigdir)