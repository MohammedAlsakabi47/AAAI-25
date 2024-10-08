import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import scipy.io
import os

# Load model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# folder of images to read
foler = 'Example Frames/camera images/'

for ii in range(10):
        print('ii: ', ii)
        input_image = Image.open(foler+str(ii+1)+'.jpg')
        input_image = input_image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize((256,256))
        r.putpalette(colors)
        r = np.where(np.array(r) != 0, 1, 0)

        # Save the NumPy array as a .mat file
        scipy.io.savemat('Example Frames/camera scemantic segmentations/'+str(ii+1)+'.mat', {'Semantic_Segmentations': r})
