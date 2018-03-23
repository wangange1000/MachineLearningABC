# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:44:51 2018

@author: wangange
"""
"""
@code: 10517391532qq.com
"""

from PIL import Image
from PIL import ImageFilter

def imageprepare(argv):
    """
    This function returns the pixel values
    The input is a png file location
    """
    
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28,28), (255)) #创建白色画布
    
    if width>height:
        nheight = int(round((20.0/width*height), 0)) #调整宽度为20
        if (nheight==0):
            nheight = 1
        #调整大小并保存
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #计算水平位置
        newImage.paste(img, (4,wtop)) #把图片贴到白画布上
    else:
        nwidth = int(round((20.0/height*width),0))
        if(nwidth==0):
            nwidth=1
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28-nwidth)/2), 0))
        newImage.paste(img, (wleft, 4))
    #newImage.save("sample.png")
    tv = list(newImage.getdata())  # 获取像素值
    # 将每个像素的值归一化
    tva = [(255-x)*1.0/255.0 for x in tv]
    return tva