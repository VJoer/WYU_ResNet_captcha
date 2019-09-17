# 五邑大学教务处子系统验证码识别

- ## 准备

        1. 下载预训练模型
            通过百度云网盘下载ResNet\model\captcha\resNet_Pretraining_model.txt中的
            预训练模型，放入该文件夹下

        2. 软件要求
            主要软件版本
            Python == 3.6
            Pytroch == 1.0.0
            CUDA == 9.0 (仅限使用GPU，如用CPU测试，则不需要)
            还有些包不再一一列出，运行缺少时自行百度pip安装

        3.
        4.
        5.

- ## 如何使用

        1. Python
            # 直接运行WYU_ResNet_captacha.py文件

            # 或在 .py文件包含以下模块
            from WYU_ResNet_captacha import *

            # 使用CPU测试
            img = Image.open("./data/test/2aap.jpg")
            print(UseCPU(img))

            # 使用GPU测试
            img = Image.open("./data/test/2aap.jpg")
            print(UseGPU(img))

        2. Java
        3. C++
        4.
        5.

- ## 结果

        1. 模型大小
                                        Totals
            Total params             11.296056M
            Trainable params         11.296056M
            Non-trainable params            0.0
            Mult-Adds             12.233777856G

        2. 准确率
            将验证码数据分为训练集(2410张)与验证集(804张)，训练模型时只使用训练集，
            测试时只使用验证集，测试准确率为98.95%

        3. GPU与CPU测试所需时间差
            具体测试时间与测试设备硬件环境有关
            CPU                           Time
            Core i7-7700HQ @ 2.80GHz × 4   ≈0.05s
            Core i7-6850K @ 3.60GHz × 12   ≈0.02s

            GPU
            在模型训练时用GPU更快，但测试单张图片不推荐使用GPU，因为与CPU差异不大

- ## 鸣谢
