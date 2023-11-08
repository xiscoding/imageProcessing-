
#8/27/22 cuda 11.7 
#if cuda? false then update the toolkit 
#or make sure you run in anaconda prompt
#conda activate tf-gpuCompatable
# pytorch error requires this: 
# https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim

    import torchvision
    from torchvision import datasets, models, transforms
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import time
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
    print("cuda?: ", torch.cuda.is_available())
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),   #must same as here
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), # data augmentation
        transforms.ToTensor(),
        #TODO: decide proper normalization
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean and std of IMAGENET
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),   #must same as here
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        #TODO: decide proper normalization
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean and std of IMAGENET
    ])

    train_dir = "C:/Users/autodrive\Desktop/cudaCnnTest/data/train"
    test_dir = "C:/Users/autodrive\Desktop/cudaCnnTest/data/test"
    train_classa_dir = "C:/Users/autodrive\Desktop/cudaCnnTest/data/train/green"
    train_classb_dir = "C:/Users/autodrive\Desktop/cudaCnnTest/data/train/red"
    train_classc_dir = "C:/Users/autodrive\Desktop/cudaCnnTest/data/train/yellow"
   
    test_classa_dir = "C:/Users/autodrive\Desktop/cudaCnnTest/data/test/green"
    test_classb_dir = "C:/Users/autodrive\Desktop/cudaCnnTest/data/test/red"
    test_classc_dir = "C:/Users/autodrive\Desktop/cudaCnnTest/data/test/yellow"
    

    train_dataset = datasets.ImageFolder(train_dir, transforms_train)
    test_dataset = datasets.ImageFolder(test_dir, transforms_test)
    # denotes the number of processes that generate batches in parallel.
    #  A high enough number of workers assures that CPU computations are efficiently managed, i.e.
    #   that the bottleneck is indeed the neural network's forward and 
    #   backward operations on the GPU (and not data generation).
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=8)

    print('Train dataset size:', len(train_dataset))
    print('Test dataset size:', len(test_dataset))
    class_names = train_dataset.classes
    print('Class names:', class_names)


    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams.update({'font.size': 20})
    def imshow(input, title):
        # torch.Tensor => numpy
        input = input.numpy().transpose((1, 2, 0))
        # undo image normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input = std * input + mean
        input = np.clip(input, 0, 1)
        # display images
        plt.imshow(input)
        plt.title(title)
        plt.show()
    # load a batch of train image
    iterator = iter(train_dataloader)
    # visualize a batch of train image
    inputs, classes = next(iterator)
    out = torchvision.utils.make_grid(inputs[:4])
    imshow(out, title=[class_names[x] for x in classes[:4]])

    model = models.resnet18(pretrained=True)   #load resnet18 model
    num_features = model.fc.in_features     #extract fc layers features
    model.fc = nn.Linear(num_features, 3) #(num_of_class == 3)
    model = model.to(device) 
    criterion = nn.CrossEntropyLoss()  #(set loss function)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 60   #(set no of epochs)
    start_time = time.time() #(for showing time)
    for epoch in range(num_epochs): #(loop for every epoch)
        print("Epoch {} running".format(epoch)) #(printing message)
        """ Training Phase """
        model.train()    #(training model)
        running_loss = 0.   #(set loss 0)
        running_corrects = 0 
        # load a batch data of images
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device) 
            print(f"labels: {labels} step#: {i}")
            # forward inputs and get output
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # get loss value and update the network weights
            loss.backward()#calculates gradients
            #TODO: CLIP GRADIENTS
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset) * 100.
        print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() -start_time))
        
        """ Testing Phase """
        model.eval()
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(test_dataset)
            epoch_acc = running_corrects / len(test_dataset) * 100.
            print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time()- start_time))

            
            
    save_path = 'resNet8_27_22.pth'
    torch.save(model.state_dict(), save_path)

    model = models.resnet18(pretrained=True)   #load resnet18 model
    num_features = model.fc.in_features #extract fc layers features
    model.fc = nn.Linear(num_features, 3)#(num_of_class == 2)
    model.load_state_dict(torch.load("resNet8_27_22.pth"))
    model.to(device)

    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams.update({'font.size': 20})
    def imshow(input, title):
        # torch.Tensor => numpy
        input = input.numpy().transpose((1, 2, 0))
        # undo image normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input = std * input + mean
        input = np.clip(input, 0, 1)
        # display images
        plt.imshow(input)
        plt.title(title)
        plt.show()

        ##Testing
    model.eval()#remove droupout, etc

    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0
        for i, (inputs, labels) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            if i == 0:
                print('======>RESULTS<======')
                images = torchvision.utils.make_grid(inputs[:4])
                imshow(images.cpu(), title=[class_names[x] for x in labels[:4]])
        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects / len(test_dataset) * 100.
        print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.
            format(epoch, epoch_loss, epoch_acc, time.time() - start_time))