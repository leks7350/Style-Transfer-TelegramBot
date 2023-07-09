import telebot
import os
import shutil
from telebot import types
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms, models
import torch.optim as optim

bot = telebot.TeleBot('6253817414:AAGNye2CFcUcNLrHQaam831HSNMCNXfaMyQ')

ListPhotos = []

if not os.path.exists('test_dir'):
    os.mkdir('test_dir')

if not os.path.exists('test_dir/results'):
    os.mkdir('test_dir/results')

if not os.path.exists('test_dir/photos'):
    os.mkdir('test_dir/photos')
else:
    shutil.rmtree('test_dir/photos/')
    os.mkdir('test_dir/photos')


@bot.message_handler(commands=['start', 'transfer_style'])
def start(message):
    markup = types.ReplyKeyboardMarkup()
    bot.send_message(message.chat.id, f'Привет, {message.from_user.first_name}! \n Я чат-бот с нейронной сетью, '
                                      f'умеющей переносить стиль с одного изображения на другое. \n '
                                      f'Пришли мне два изоражения: '
                                      f'\n 1ое - которое будет преобразовано. '
                                      f'\n 2ое - стиль которого будет перенесен на 1ое изображение.', reply_markup=markup)


@bot.message_handler(content_types=['photo'])
def reply(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = 'test_dir/' + file_info.file_path
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    ListPhotos.append(src)

    if len(ListPhotos) == 1:
        bot.reply_to(message, f'Это будет основным изображением. \n Жду второе...')
    elif len(ListPhotos) == 2:
        bot.reply_to(message, f'В стиль этого изображения будет преобразовано предыдущее.')
    else:
        bot.send_message(message.chat.id, f'Send me photo one-by-one. Try again')
        shutil.rmtree('test_dir/photos/')
        os.mkdir('test_dir/photos')
        ListPhotos.clear()


    if len(ListPhotos) == 2:
        bot.send_message(message.chat.id, f'Отлично! Теперь у меня есть оба изображения! '
                                          f'\n Через минуту я вернусь с результатом...')

        img = start_style_transfer(ListPhotos)
        ListPhotos.clear()

        bot.send_photo(message.chat.id, photo=open(img, 'rb'))



def start_style_transfer(imagelist):
    content = load_image(imagelist[0]).to(device)
    style = load_image(imagelist[1]).to(device)

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    target = content.clone().requires_grad_(True).to(device)

    style_weights = {'conv1_1': 1,
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}
    content_weight = 1  # alpha
    style_weight = 1e3  # beta

    optimizer = optim.Adam([target], lr=0.003)
    steps = 5000

    for ii in range(1, steps + 1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        style_loss = 0

        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss / (d * h * w)
        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    #собираем имя результата обработки
    counter_name = len(os.listdir('test_dir/results/'))
    result_image = f'test_dir/results/result_{counter_name}.png'
    #сохраняем результат
    plt.axis('off')
    plt.imshow(im_convert(target))
    plt.savefig(result_image)
    plt.close()

    return result_image



vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)






def load_image(img_path, max_size=300):
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
    features = {}
    x = image

    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


bot.infinity_polling()
