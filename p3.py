from pdb import set_trace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from matplotlib.pyplot import imshow

# JSMA method
class JSMA():
    def __init__(self, width, height, channel, cmin, cmax, n_class) -> None:
        self.width = width
        self.height = height
        self.channel = channel
        self.input_size = width * height * channel
        self.cmin = cmin
        self.cmax = cmax
        self.n_class = n_class

    def saliency_map(self, phi, dtdx, dodx):
        """
        Saliency map function that returns score for each input dimension.
        Algorithm 3 Increasing pixel intensities saliency map
        """
        set_trace()
        _max = 0
        
        for pixel_pair in phi:
            _alpha = torch.sum(dtdx[pixel_pair[0,0], pixel_pair[0,1], :] + dtdx[pixel_pair[1,0], pixel_pair[1,1], :])
            if _alpha <= 0:
                continue

            _beta = torch.sum(dodx[pixel_pair[0,0], pixel_pair[0,1], :] + dodx[pixel_pair[1,0], pixel_pair[1,1], :])
            if _beta >= 0:
                continue

            if (-_alpha * _beta > _max): 
                selected_pixel_pair = pixel_pair
                _max = -_alpha * _beta

        return selected_pixel_pair

    def jacobian_matrix(self, my_nn_model, x, n_class):
        """
        Calculate jacobian of logits wrt input.
        """
        inp = x.detach().clone()
        Jn = torch.zeros((self.width, self.height, n_class))  # loop will fill in Jacobian
        Jn = Jn.float()

        inp.requires_grad_()

        preds = my_nn_model(inp)
        for i in range(n_class):
            grd = torch.zeros((1, n_class))
            #.cuda()  # same shape as preds
            grd[0, i] = 1  # column of Jacobian to compute
            preds.backward(gradient=grd, retain_graph=True)
            Jn[:, :, i] = inp.grad.float()  # fill in one column of Jacobian
            inp.grad.zero_()  # .backward() accumulates gradients, so reset to zero

        return Jn

    def jsma(self, phi, X_adv, target_y, model, eps, cmin=0.0, cmax=1.0):
        """
        Implementation of JSMA method to generate adversarial images.
        """
        set_trace()
        # Get model logits and probs for the input.
        # logits, probs = model(torch.reshape(X_adv, shape=(-1, self.width, self.height, self.channel)))
        probs = model(X_adv)
        
        # Get model prediction for inputs.
        y_ind = torch.argmax(probs[0])
        print(probs[0])
        
        import time;start = time.time()
        # Calculate jacobian matrix of logits wrt to input.
        jacobian = self.jacobian_matrix(model, X_adv, self.n_class)
        end = time.time();print("Calculate jacobian matrix of logits wrt to input {}".format(end - start))
        
        grad_target = jacobian[:, :, target_y]
        
        mask_grad_other = torch.ones(self.n_class)
        mask_grad_other[grad_target.long()] = 0
        grad_other = jacobian[:, :, mask_grad_other==1]
        
        start = time.time()
        pixel_pair = self.saliency_map(phi, grad_target, grad_other)
        print(pixel_pair)
        end = time.time();print("Compute saliency score for each dimension {}".format(end - start))

        # perturb the input image X
        X_adv[0, 0, pixel_pair[0,0], pixel_pair[0,1]] += eps
        X_adv[0, 0, pixel_pair[1,0], pixel_pair[1,1]] += eps

        start = time.time()

        # remove the pixel pair whose values are out of the [cmin, cmax]
        update_phi = []
        c1 = (eps < 0 and X_adv[0, 0, pixel_pair[0,0], pixel_pair[0,1]] <= cmin) or (eps > 0 and X_adv[0, 0, pixel_pair[0,0], pixel_pair[0,1]] >= cmax)
        c2 =(eps < 0 and X_adv[0, 0, pixel_pair[1,0], pixel_pair[1,1]] <= cmin) or (eps > 0 and X_adv[0, 0, pixel_pair[1,0], pixel_pair[1,1]] >= cmax)
        if c1 and c2:
            for _item in phi:
                if torch.equal(pixel_pair[0], _item[0]) or torch.equal(pixel_pair[0], _item[1]):
                    continue
                if torch.equal(pixel_pair[1], _item[0]) or torch.equal(pixel_pair[1], _item[1]):
                    continue
                update_phi.append(_item)
            phi = update_phi
            phi = torch.stack(phi)
                                        
        elif c1:
            for _item in phi:
                if torch.equal(pixel_pair[0], _item[0]) or torch.equal(pixel_pair[0], _item[1]):
                    continue
                update_phi.append(_item)
            phi = update_phi
            phi = torch.stack(phi)

        elif c2:
            for _item in phi:
                if torch.equal(pixel_pair[1], _item[0]) or torch.equal(pixel_pair[1], _item[1]):
                    continue
                update_phi.append(_item)
            phi = update_phi
            phi = torch.stack(phi)

        else:
            pass  # no pixel pair needs to be removed 
        
        end = time.time();print("update_phi {}".format(end - start))

        X_adv = torch.clamp(X_adv, cmin, cmax)
        print(X_adv[0, 0, pixel_pair[0,0], pixel_pair[0,1]])
        print(X_adv[0, 0, pixel_pair[1,0], pixel_pair[1,1]])

        return X_adv, y_ind, phi

    def generate_jsma(self, model, X, target, eps=1.0/255, epochs=50):
        """
        Run JSMA on input image for `epochs` number of times.
        """
        set_trace()
        torch.manual_seed(42)

        probs = model(X)
        y_ind = torch.argmax(probs[0])
        pert_X = X.clone()

        # generate the initial pixel pair set
        temp_phi = []
        phi = []
        for i in range(self.width):
            for j in range(self.height):
                temp_phi.append([i, j])
        
        for i_phi in temp_phi:
            for j_phi in temp_phi:
                if i_phi != j_phi:
                    phi.append([i_phi, j_phi])
       
        set_trace()
 
        len_temp_phi = len(temp_phi) * (len(temp_phi)-1)
        
        phi = phi[:int(len_temp_phi/2)]
        phi = torch.Tensor(phi).long()

        # Op for one iteration of jsma.
        _epoch = 0
        while not (_epoch >= epochs or y_ind == target or phi == []):
            pert_X, y_ind, phi = self.jsma(phi, pert_X, target_y=target, model=model, eps=eps)
            print("generate_jsma epochs: {}".format(_epoch))
            _epoch+=1
    
        pert = pert_X - X
            
        return pert_X.reshape(-1, self.width, self.height, self.channel), pert.reshape(-1, self.width, self.height, self.channel), phi

# CNN model
class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1, padding_mode='zeros')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1, padding_mode='zeros')
        self.fc1 = nn.Linear(7 * 7 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# LeNet Model definition
class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

set_trace()

# load torch pre-trained MNIST model and dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
batch_size = 600
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=False)

# select an image of training set with label 1
print(train_data[3][1])

# Train or load the existing the CNN model
PATH = './mnist_net.pth'
PATH = './model.pth'
net = CNN_model()

set_trace()

if os.path.exists(PATH):
    net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))   
    print('Finished loading') 
else:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1000):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #inputs = inputs.cuda()
            #labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0
    
    torch.save(net.state_dict(), PATH)
    print('Finished Training')

set_trace()

# test overall accuracy
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        #images = images.cuda()
        #labels = labels.cuda()
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

set_trace()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# test accuracy for each class
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        #images = images.cuda()
        #labels = labels.cuda()
        
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

set_trace()

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

set_trace()

# launch JSMA attack on CNN model
# init hyper-parameter
width = 28
height = 28
channel = 1
cmin = 0.0
cmax = 1.0
n_class = 10

# the images with label 1
images = torch.Tensor(train_data[3][0][None])
#.cuda()

# the targeted label of perturbation
target = torch.Tensor([7]).long()
#.cuda()
print("Original label: {}".format(train_data[3][1]))
print("Targeted label: {}".format(target))

jsma_attack = JSMA(width, height, channel, cmin, cmax, n_class)

# the epsilon > 0
pert_X, pert, phi = jsma_attack.generate_jsma(net, images, target, eps=1./255, epochs=2000)

# plot the perturbed image
imshow(pert_X[0].cpu())

# only the perturbation
imshow(pert[0].cpu())
