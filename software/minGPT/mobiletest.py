import torch 
from torchvision import transforms
import sys
from PIL import Image
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()
chartset="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456790";
charsetLen = len(chartset)
step=2/charsetLen

def get_logits(path : str ):
    input_image = Image.open(path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    out=model(input_batch)
    #out=torch.nn.functional.normalize(out)
    offset = torch.abs(torch.min(out[0]))
    print(out.shape)
    return torch.floor(out[0]+offset)


a = get_logits(sys.argv[1]);
b = get_logits(sys.argv[2]);
sim=torch.dot(a,b)
a=a.tolist()
a="".join(list(map(lambda x : chartset[int(x)],a)))
print(a)
