from torchvision import transforms

T_G_WIDTH = 100
# T_G_HEIGHT = 224
T_G_NUMCHANNELS = 3
T_G_SEED = 1337


# Image Transforms for pre-trained model.
# Normalization parameters taken from documentation for pre-trained model.
input_size = T_G_WIDTH
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
