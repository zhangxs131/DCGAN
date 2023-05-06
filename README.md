# DCGAN

本项目为学习DCGAN编写，来源于pytorch tutorials https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

main.py是一个训练测试例子，里面包括数据集读取，生成器，辨别器的定义，以及整个训练步骤

    使用的数据为人脸的数据集，地址为 http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    其Google Drive 为 https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ

    下载其中的img_align_celeba.zip，并将解压后的文件img_align_celeba放入Data/celeba中

    所有使用参数均可以在def main()函数中调整

generate.py 是使用训练好的生成器模型netG.pth进行生成，并显示图片的函数

netG.pth netD.pth 分别是训练10epoch得到的模型，效果一般可以多训练几轮，10个epoch差不多单gpu训练了15分钟左右

formal_main.py 是正式版的DCGAN教程代码，支持使用多个数据集以及自己的数据集。
代码源地址： https://github.com/pytorch/examples/tree/master/dcgan
