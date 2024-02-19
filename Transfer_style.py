import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image


device= torch.device("cuda"if torch.cuda.is_available else "cpu")
image_size = 365

epoches= 10000
learning_rate= 0.001
alpha= 1
beta=0.01


class VGG(nn.Module):
  def __init__(self) :
    super(VGG,self).__init__()
    #we are chossing the first conv layer after the max pooling
    self.chosen_features = ['0','5','10','19','28']
    #we are slelcting only upto 28 layers(29-1) for our model
    self.model= models.vgg19(pretrained=True).features[:29]

  def forward(self,x):
    #getting chosen feature layers from model
    features=[]
    for layer_num,layer in enumerate(self.model):
      x=layer(x)
      if str(layer_num) in self.chosen_features:
        features.append(x)
    return features


#we are loading unsquzeeing the image and moving it to device ie. GPU or CPU
def load_image(image_name):
  image=Image.open(image_name)
  image=loader(image).unsqueeze(0)
  return image.to (device)

loader=transforms.Compose(
  [
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor()
    ]
    )


original_image=load_image("mona_lisa.PNG")
style_image=load_image("hex.jpg")


model=VGG().to(device).eval()
generated=original_image.clone().requires_grad_(True)


#setting optimizer
optimizer=optim.Adam([generated],lr=learning_rate)


#training loop
for steps in range(epoches):
  generated_features=model(generated)
  original_features=model(original_image)
  style_features=model(style_image)

  style_loss = original_loss = 0

  for gen_features,orgin_features,sty_features in zip(generated_features,original_features,style_features):
    batch_size,channel,height,width=gen_features.shape
    original_loss += torch.mean((gen_features-orgin_features)**2)

    #compute gram matrix
    G = gen_features.view(channel,height*width).mm(gen_features.view(channel,height*width).t())
    
    A = sty_features.view(channel,height*width).mm(sty_features.view(channel,height*width).t())

    style_loss += torch.mean((G-A)**2)


  total_loss = alpha * original_loss + beta * style_loss
  optimizer.zero_grad()
  total_loss.backward()
  optimizer.step()

  if steps % 100 == 0:
    print(total_loss)
    save_image(generated,"genarated.png")