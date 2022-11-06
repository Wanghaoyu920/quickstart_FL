
import  torchvision.models as models
def get_model()->models.vgg.VGG:
    vgg19_model = models.vgg19(pretrained=False) ## 就会自动下载vgg19的参数文件并放在本地缓存中。所以不用提供本地参数文件的路径
    print(type(vgg19_model))
    return vgg19_model



