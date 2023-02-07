import numpy as np
import cv2
import torch
import glob as glob
import torchvision
import warnings
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

def calc_inference(original, model):           #same type given from cv2.imread()
    all_boxes = []
    # classes: 0 index is reserved for background
    CLASSES = [
        'background', 'compOp', 'numPad', 'baseOp', 'outputbox'
    ]
    # define the detection threshold...
    # ... any detection having score below this will be discarded
    detection_threshold = 0.9
    
    img = cv2.resize(original,(512,512))
    image = img.astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float64)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)
        
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # draw the bounding boxes and write the class name on top of it         *1080 / 512
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(original,
                        (int(box[0]*1080 / 512), int(box[1]*1080 / 512)),
                        (int(box[2]*1080 / 512), int(box[3]*1080 / 512)),
                        (0, 0, 255), 2)
            cv2.putText(original, pred_classes[j], 
                        (int(box[0]*1080 / 512), int(box[1]*1080 / 512 -5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                        2, lineType=cv2.LINE_AA)
            all_boxes.append(((box[0],box[1],box[2],box[3],pred_classes[j])))
        cv2.imshow('Prediction', original)
        cv2.waitKey(0)
        cv2.destroyWindow('Prediction')
    

if __name__ == "__main__":
    allBoxes=[]
    warnings.filterwarnings("ignore")
    screenWidth = 24.9 
    screenHeight = 16.8 
    ardCon = False  #flag to check serial connection

    # detection model
    # set the computation device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model and the trained weights
    calcModel = create_model(num_classes=5).to(device)      
    calcModel.load_state_dict(torch.load(
        'C:\\Users\\nicco\\Documents\\PlatformIO\\Projects\ScreenAI\\models\\calcmodel50.pth', map_location=device    #TODO: put your model
    ))
    calcModel.eval()

    #start capture and inference on it
    image = hwp.getframe()
    (h,w) = image.shape[:2]
    scaled_img=image[0:1080, 420:1500]
    if result:
        allBoxes = calc_inference(scaled_img,calcModel)
    else:
        print("inference error")
    flagCalcCheck = [False,False,False,False]
    button = {
        "0":[],
        "1":[],
        "2":[],
        "3":[],
        "4":[],
        "5":[],
        "6":[],
        "7":[],
        "8":[],
        "9":[],
        "0":[],
        ".":[],
        "/":[],
        "x":[],
        "+":[],
        "-":[],
        "=":[]
    }
    for box in allBoxes:
        if box[4] == "numPad":
            flagCalcCheck[0]=True
            width = (box[2]-box[0])/3
            height = (box[3]-box[1])/4
            button["0"]=[box[0],        box[1]+3*height,box[2]+2*width, box[3]+4*height]
            button["1"]=[box[0],        box[1]+2*height,box[2]+width,   box[3]+3*height]
            button["2"]=[box[0]+width,  box[1]+2*height,box[2]+2*width, box[3]+3*height]
            button["3"]=[box[0]+2*width,box[1]+2*height,box[2]+3*width, box[3]+3*height]
            button["4"]=[box[0],        box[1]+height,  box[2]+width,   box[3]+2*height]
            button["5"]=[box[0]+width,  box[1]+height,  box[2]+2*width, box[3]+2*height]
            button["6"]=[box[0]+2*width,box[1]+height,  box[2]+3*width, box[3]+2*height]
            button["7"]=[box[0],        box[1],         box[2]+width,   box[3]+height]
            button["8"]=[box[0]+width,  box[1],         box[2]+2*width, box[3]+height]
            button["9"]=[box[0]+2*width,box[1],         box[2]+3*width, box[3]+height]
            button["."]=[box[0]+2*width,box[1]+3*height,box[2]+3*width, box[3]+4*height]
            continue
        if box[4] == "baseOp":
            flagCalcCheck[1]=True
            width = (box[2]-box[0])/2
            height = (box[3]-box[1])/4
            button["/"]=[box[0],        box[1],         box[2]+width,   box[3]+height]
            button["x"]=[box[0],        box[1]+height,  box[2]+width,   box[3]+2*height]
            button["-"]=[box[0],        box[1]+2*height,box[2]+width,   box[3]+3*height]
            button["+"]=[box[0],        box[1]+3*height,box[2]+width,   box[3]+4*height]
            button["("]=[box[0]+width,  box[1],         box[2]+2*width, box[3]+height]
            button[")"]=[box[0]+width,  box[1]+height,  box[2]+2*width, box[3]+2*height]
            button["="]=[box[0]+width,  box[1]+2*height,box[2]+2*width, box[3]+4*height]
        if box[4] == "compOP":
            flagCalcCheck[2]=True
        if box[4] == "outputbox":
            flagCalcCheck[3]=True
