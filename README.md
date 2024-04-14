# Deep Learning Step By Step
================

## Overview

This is a summary of projects that I developed for study of Deep Learning.

There are many frameworks and libraries out there. Fastai is a good starting point. Huggingface has huge amount of models for use. From scratch projects are very good for learning the fundamentals. 

There are so many ML concepts, designs and models to learn and try. I hope to grasp this revolutionary technology and make use of AI to solve real world problems. 

## Projects that have been completed

1. <b>`Pet classifier`</b>

    This is a fine-tuned resnet that can do simple image classification (cat or dog) task.
    
    Try it at Huggingface Spaces: [A pet classifier](https://huggingface.co/spaces/wb-droid/pet_classifier)

    References:

    The original model definition and training can be found in fastai's ["Saving a Cats v Dogs Model"](https://github.com/fastai/course22/blob/master/02-saving-a-basic-fastai-model.ipynb).<br>
    The method to build a gradio web app on Huggingface Spaces is nicely described in Tanishq's ["Gradio + HuggingFace Spaces: A Tutorial"](https://www.tanishq.ai/blog/posts/2021-11-16-gradio-huggingface.html).

2. <b>`HDB price predictor`</b>

    There are some very good existing HDB price predictor projects using RandomForests. I tried something different -- training a neural network to predict HDB prices. I only traind with a small portion of old data from 1997-01 to 2020-01 for evaluation. The result is pretty good with <b>mean absolute error of S$14k</b>. For prediction after training period, the mean absolute error is S$20k. When using this model to predict current price at 2024-03, after adjust by Resale Price Index, the predicted price has a bit higher mean absolute error of S$37k.

    Try the model at Huggingface Spaces: [HDB price predictor](https://huggingface.co/spaces/wb-droid/HDB_price_predictor).

    Data analysis, model, training and evaulation details can be found in this [Jupyter notebook](HDB_price_predictor/PricePredictor.ipynb).

    References:

    This model uses [fastai's framework](https://docs.fast.ai/tutorial.tabular.html).<br>
    There is a very good [Exploratory Data Analysis on the HDB data by teyang-lau](https://github.com/teyang-lau/HDB_Resale_Prices). His model was built with Linear Regression and RandomForest.<br>
    Data is from [data.gov.sg](https://data.gov.sg).

3. <b>`Stable Diffusion Text Inversion`</b>

    Stable Diffusion can be fine-tuned to add new concept/style. This is an interesting scheme. 
    I trained Stable Diffusion to add a new token `<monkey-toy>` that represents my son's monkey toy: 
    
    <img src="Text_inversion/my_concept/0.jpg" width="200">
        
    After that, prompt of `"a <monkey-toy> in space suit"` can generate this: 
    
    <img src="Text_inversion/generated/003.png" width="200">

    Data, model, training and inference details can be found in this [Jupyter notebook](Text_inversion/sd_textual_inversion_training_mk.ipynb). 

    References:

    This model and training code is mainly from [Huggingface's notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb).<br>
    Additonal info can be found in [huggingface](https://huggingface.co/docs/diffusers/en/using-diffusers/loading_adapters#textual-inversion).

4. <b>`Image Captioning`</b>

    Image captioning is more flexible then categorization, and privides more infomation on the objects in the image such as positional data. COCA Model claimed to have have the capability "to directly applicable to image captioning tasks as an encoder-decoder model". I tested the "monkey toy" image  
    <img src="Text_inversion/my_concept/0.jpg" width="32"> with OpenCLIP and COCA. The generated caption is "a stuffed monkey with a crown on it 's head". Pretty good.

    Model and inference details can be found in this [Jupyter notebook](Image_caption/openCLIP_COCA_image_caption.ipynb). 

    References:

    The code is from this [OpenCLIP's notebook](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb). <br>
    CoCa paper is [here](https://arxiv.org/abs/2205.01917).<br>
    OpenCLIP repo is [here](https://github.com/mlfoundations/open_clip).
