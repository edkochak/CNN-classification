from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T
from torchvision import models
import torchvision
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

dataset_by_model ={
    '6.model': 'test_small',
    '8.model': 'test'
}





def get_num_correct(preds, labels):
    # Количество правильных предиктов
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_data_loader(folder, batch_size=64):
    # Создаем датасет
    data_dir = os.path.join(os.getcwd(), folder)
    transform = T.Compose([
        T.CenterCrop((150,150)),
        T.Resize([47, 47]), 
        
        
        T.ToTensor()
    ])

    data = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset=data, batch_size=batch_size, shuffle=True)

    return data_loader


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18(pretrained=True)

        self.model.fc = nn.Sequential(nn.Linear(
            self.model.fc.in_features, 3000), nn.ReLU(), nn.Linear(3000, num_classes))
        # Изменяем последний слой под нашу классификацию

    def forward(self, x):
        x = self.model(x)
        return x



def check_on_dataset(model_name):
    model = Net(len(names))
    model.load_state_dict(torch.load(model_name))
    model.eval()
    n = 2
    total_correct = 0
    for _ in range(n):
        tests = get_data_loader(dataset_by_model[model_name])
        for imgs_batch, answers_batch in tests:
            out = model(imgs_batch)
        
            check_predicts = get_num_correct(out, answers_batch)
            
            total_correct += check_predicts
    
    return total_correct/(len(tests.dataset)*n)


def check_for_img_from_internet(path,model_name):
    transform = T.Compose([
        T.Resize([47, 47]),
        T.ToTensor()
    ])

    model = Net(len(names))
    model.load_state_dict(torch.load(model_name))
    model.eval()
    d = 0.2
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    imgs = torch.Tensor()
    batch_size = 64
    files = os.listdir(path)

    for file in files:
        original_image = cv2.imread(os.path.join(path,file))
        h1, w1, _ = original_image.shape
        grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(grayscale_image)

        

        for (column, row, width, height) in detected_faces:
            dx = int(d*width)
            dy = int(d*height)

            face_image = original_image[max(row-dy,0):min(row +
                                            height+dy,h1), max(column-dx,0):min(column+width+dx,w1),::-1]
            
            img = Image.fromarray(face_image)
            img = transform(img).unsqueeze(0)

            if imgs.shape[0] == 0:
                imgs = img
            else:
                imgs = torch.cat((imgs,img))
    img_count = imgs.shape[0]
    loop_count = int(np.ceil(img_count/batch_size))
    all_answers = []
    soft = nn.Softmax(dim=1)
    count = 0

    for i_loop in range(loop_count):
        predicts = model(imgs[i_loop*batch_size:(i_loop+1)*batch_size])
        predicts = soft(predicts)
        for predict in predicts:
            
            _temp = files[count]
            topk =torch.topk(predict,3)
            for value,indice in zip(topk.values,topk.indices):
                if value.item()>0.1:
                    _temp+=f'\n {int(value.item()*100)}% - {names[indice.item()]}'
            all_answers.append(_temp)

            count+=1
    print('\n\n'.join(all_answers))
                

    
    



def get_names(dataset_name):
    data = get_data_loader(dataset_name)
    names = {}
    for path, index in data.dataset.samples:
        if index not in names:
            name = path.split(f'{dataset_name}\\')[-1].split('\\')[0]
            names[index] = name
    return names



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN классификатор')
    parser.add_argument("--show_names", default=0, help="Отображает доступные классы. Значения 1/0")
    parser.add_argument("--model", default="8.model", help="Путь до модели, 8.model - 182 класса, 6.model - 12 классов.")
    parser.add_argument("--img_path", default='img_from_internet', help="Путь до папки с изображениями, которые надо классифицировать.")
    args = parser.parse_args()


    
    path_to_images_from_internet = args.img_path
    model_name = args.model

    names = get_names(dataset_by_model[model_name])
    if int(args.show_names):
        print('\n'.join(names.values()))
    print('Всего людей: ',len(names))
    print(f'Проверка на тестах: {check_on_dataset(model_name)}')
    print('Фотографии из интернета:')
    check_for_img_from_internet(path_to_images_from_internet,model_name)
    
