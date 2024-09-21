# Import here 
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.nn import nn
from torch.optim import optim
from torch.util.data import DataLoader

import requests 
import argparse
import json
from PIL import Images
from collections import OrderedDict
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a new Classifier 
model = models.vggs(pretrained=True)

for param in model.parameters():
    param.require_grad=False
    
# Function to load the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
#     model.input = checkpoint['input_size']
#     model.output = checkpoint['output_size']
    model.epochs = checkpoint['epochs']
    model.classifier = Classifier(25088, 4096, 102)
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    dim_adjust = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])
                            ])
    pil_image = dim_adjust(pil_image)
    np_image = np.array(pil_image)
    return np_image


def predict(image_path, model, topk=5):
    # Load and preprocess the image
    image = process_image(image_path)
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Move image to the appropriate device
    image = image.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        output = model(image)
    
     # Get probabilities and class indices
    probs, indices = torch.topk(torch.nn.functional.softmax(output, dim=1), topk)
    probs = probs.squeeze().cpu().numpy()
    indices = indices.squeeze().cpu().numpy()
    
    # Map indices to class names
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]
    
    return probs, classes


parser = argparse.ArgumentParser(description= 'Predicting for image Classifier')
parser.add_argument('input_img',default='./flowers/test/13/image_05775.jpg',action='store')
parser.add_argument('--checkpoint',default='./checkpoint.pth',action='store')
parser.add_argument('--top_k',type=int,default=5,action='store')
parser.add_argument('--cat_to_name',default='cat_to_name.json',action='store')
parser.add_argument('--gpu',default='gpu',action='store')

input_argu = parser.parse_args()
input_img = input_argu.input_img
checkpoint = input_argu.checkpoint
topk =input_argu.top_k
cat_to_name = input_argu.cat_to_name
device = input_argu.gpu

def main():
    model=load_checkpoint(checkpoint)
    device = input_argu.gpu
    model.to(device)
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    probs, classes = predict('./flowers/test/13/image_05775.jpg',model)
    flower_names = [cat_to_name.get(i) for i in classes]   
    predicts = {name: prob for name, prob in zip(flower_names, probs)}
    print(predicts)

if __name__ == "__main__":
    main()



