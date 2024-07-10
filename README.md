# Deep Learning Step By Step
================

## Overview

This is a summary of projects that I developed for study of Deep Learning.

There are many frameworks and libraries out there. Fastai is a good starting point. Huggingface has huge amount of models for use. From scratch projects are very good for learning the fundamentals. 

There are so many ML concepts, designs and models to learn and try. I hope to grasp this revolutionary technology and make use of AI to solve real world problems. 

## From Scratch Models/Applications

1. <b>`HDB price predictor`</b>

    There are some very good existing HDB price predictor projects using RandomForests. I tried something different -- training a neural network to predict HDB prices. I only traind with a small portion of old data from 1997-01 to 2020-01 for evaluation. The result is pretty good with <b>mean absolute error of S$14k</b>. For prediction after training period, the mean absolute error is S$20k. When using this model to predict current price at 2024-03, after adjust by Resale Price Index, the predicted price has a bit higher mean absolute error of S$37k.

    Try the model at Huggingface Spaces: [HDB price predictor](https://huggingface.co/spaces/wb-droid/HDB_price_predictor).
    <br><img src="HDB_price_predictor/screenshot/0.jpg" width="240">

    Data analysis, model, training and evaulation details can be found in this [Jupyter notebook](HDB_price_predictor/PricePredictor.ipynb).

    References:

    This model uses [fastai's framework](https://docs.fast.ai/tutorial.tabular.html).<br>
    There is a very good [Exploratory Data Analysis on the HDB data by teyang-lau](https://github.com/teyang-lau/HDB_Resale_Prices). His model was built with Linear Regression and RandomForest.<br>
    Data is from [data.gov.sg](https://data.gov.sg).

2. <b>`Chinese poem generator`</b>

    NLP Transformer models are very popular. There are many existing implementations and pretrained models. Creating one from scratch is good learning and practise. To make it managable and useful, a Chinese poem generator is the target.

    a) A basic MLP model from scratch.

    Data is from https://github.com/Werneror/Poetry.

    First created a Chinese character level tokenizer. Built a simple Model just with token embedding, position embedding and MLP. 
    
    Modle/training details can be found [here](https://github.com/wb-droid/myGPT/blob/main/MLP_model/GPTModel_MLP_.ipynb).

    After training for a while, the following can be generated.

    `generate('终南')` -- `'终南汞懒飞收。俗始闻夜门。谁常波漫春'`
    
    `generate('灵者')` -- `'灵者轩愁。月看曲，贱朱光受，书初去雨'` 


    b) A GPT2-like model from scratch.

    Follow the original GPT2 design and "Attention is all you need" paper. Add on top of the above MLP model to have all the additional critical components such as casual self attention, layer norm, dropout, skip connection, etc. After training for a while, the following nice poems can be generated.

    `generate('终南')` -- `'终南岸，水碧绿槐无处色。云雨初寒落月，江风满天。秋景遥，夜深烟暮春。一望青山里，千嶂孤城下，何远近东流。古人不见长空愁，万般心生泪难尽。'`
    
    `generate('灵者')` -- `'灵者，寒暑气凝空濛长。风雨如霜月，万顷不闻钟客归。白纻初行尽柳枝，黄花满衣无愁懒。春色，残红芳兰深，一声兮，九陌上，相思君王。'` 

    Model and training details can be found in https://github.com/wb-droid/myGPT/tree/main/GPT2_like.

    Try it at Huggingface Space: https://huggingface.co/spaces/wb-droid/myGPT.
    <br><img src="myGPT/screenshot/0.jpg" width="640">
    <br><img src="myGPT/screenshot/1.jpg" width="640">

3. <b>`Sentence Embedding, Vector Database and Semantic Search`</b>

    Besides used in semantic search applications, this project is also a good foundation for Retrival Augmented Generation I plan to build later.
    This is also very similar to CLIP model that performs contrastive training on image and text embedding. 

    a) Build a text embedding model with "BERT + mean pooling". 

    b) Build a contrastive training model following the paper [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084). Train it to improve semantic search accuracy. 
    <br><img src="myTextEmbedding/training_model.jpg" width="240">

    c) Then use the trained model to implement a vector database (using data from a few wiki pages, chunk by sentence). Support cosine similarity for similarity search. 
    
    When testing with the model untrained, the search is not accurate. 
    
    `search_document("what is BERT?")` returns `Research in similarity search is dominated by the inherent problems of searching over complex objects.`, etc.

    When testing with the trained model, the search result is improved a lot.

    `search_document("what is BERT?")` returns `In practice however, BERT's sentence embedding with the [CLS] token achieves poor performance, often worse than simply averaging non-contextual word embeddings.`, etc.

    More details on the model, training and inference can be found [here](myTextEmbedding/).

    d) This model can be made faster and lighter. Embedding dimension reduction can help to reduce embedding storage but does not help model memory. Quantization is already used before on ChatGLM. So, this time I will try model Distillation, which uses the bigger model (teacher) to train a smaller model (student). Ilustrated below.  
    <br><img src="myTextEmbedding/distillation.png" width="240"> [This diagram source](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/monolingual-distillation.png) is from sentence-transformers.
    
    The student model is built by reducing BertEncoder from 12 BertLayers to 6 layers. And the model size is almost halved from 430MB to 260MB. Training model is implement as the diagram. After training, the student model performs similarly as the teacher, with matching top-2 similarity search result. More details on the models, training and inference can be found [here](./myTextEmbedding/model_distillation.ipynb).

    e) Build a huggingface space app to demo this model. An example vector database is pre-built with concepts searched from wiki, by [this script](./myTextEmbedding/create_vector_table.ipynb). User enters a question related to the concept. The app will use the question to do semantic search in the vector database and return the result.

    Try it at Huggingface Space [here](https://huggingface.co/spaces/wb-droid/SentenceEmbedding).
    <br><img src="./myTextEmbedding/application_example.jpg" width="400">
    <img src="./myTextEmbedding/application_example2.jpg" width="400">

4. <b>`Spelling Tester App`</b>

    I often need to help my son prepare his school spelling test, reading the words/dictations to him multiple time. This is a task better done by AI to save me some time and to have better pronunciation than me. My son can also practise when I'm not around. 

    a) The main component is text-to-voice. I evaluated [bark](https://github.com/suno-ai/bark) first. It's a GPT-style transfomer model similar to the myGPT model I did above. The voice generation procedure and result can be found [here](./SpellingTester/text2voice_bark.ipynb). The generated voice is pretty good. It requires a lot of GPU VRAM resources and take a bit of time to generate. The voice generated is natural but not the studio-quality.

    I also tested [edge-tts](https://github.com/rany2/edge-tts). It uses Microsoft Edge's online text-to-speech service, so the model behind is not clear to me. The voice generation procedure and result can be found [here](./SpellingTester/text2voice_edge.ipynb). The generated voice is of better quality but not as natural. It does not use local resource and generates faster. But it needs internet access, and depends on the Microsoft service which may not be available/free forever. 
    
    My son is happy with the mp3 generated as he can pause, rewind or adjust speed in the mp3 player. He now uses it to prepare for the next Chinese spelling test by himself. 

    b) Next is to use OCR to convert spelling list scan image to text, to save effort of typing. Optical Character Recognition (OCR) is mature with high accuracy. There are many open-source solutions. 
    
    I first tested [EasyOCR](https://github.com/JaidedAI/EasyOCR). Its pipeline make use of several models, such as CRAFT for detection, Resnet for feature extraction, LSTM for sequence labeling and CTC for decoding. The detection result has some errors as shown [here](./SpellingTester/img2txt_EasyOCR.ipynb).
    
    Next I tested [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). Its pipeline uses even more [models](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/ppocr_v4/ppocrv4_framework.png). The result is much better as shown [here](./SpellingTester/img2txt_paddle.ipynb). All Chinese characters and English letters are recognized correctly. So I will choose PaddleOCR.
    
    c) Create an app to do everying together: image-to-text, text-to-voice, and play the generated audio to do mock test. 
    Try it at Huggingface Space [here](https://huggingface.co/spaces/wb-droid/SpellingTester).    
    <br><img src="./SpellingTester/app_example_ch.jpg" height="150">
    <img src="./SpellingTester/app_example_en.jpg" height="150">

5. <b>`Composition Corrector App`</b>

    Composition is another area my son needs extra help. But I'm not good at it either. LLM is already very advanced and should do a better job than me. This "Composition Corrector app" is designed to find mistakes, suggest corrections and make improvements. 

    a) LLM setup

    The easier way to use a good LLM in application is to use ChatGPT API. But it's not free and not opensource. 

    Another way is to run LLM like llama and ChatGLM locally. With my small GPU, this is still doable after extra efforts as shown in the "model quantization" section below. 

    But I'm choosing a third way -- Huggingface's Inference API. It's same effect as running pretrained models locally, but it is free, more flexible and easier to maintain. 
    
    The app will use the latest and greatest [Llama 3](https://llama.meta.com/) as the language model.

    b) Prompt Engineering

    System prompt is "You are a helpful and honest assistant. Please, respond concisely and truthfully."

    I do a search of error sentence first, with this prompt -- "Only answer Yes or No. Is there grammatical or logical mistake in the sentence:".

    Then, the detail of the mistakes and correction is queried by this prompt -- "What is the mistake and what is the correct sentence?"

    c) Create the App to put everything together: image-to-text, text to query LLM to find mistake and get correction. Indeed it can find more mistakes than me. And the suggested corrections are better than what I usually come up with. Also, after making corrections, make sure to run through the Corrector again. It may still detect a few extra errors that are more subtle. 

    Try it at Huggingface Space [here](https://huggingface.co/spaces/wb-droid/CompositionCorrector).
    <br><img src="./CompositionCorrector/app_example.jpg" height="300">

6. <b>`Retrieval Augmented Generation App`</b>

    LLM can learn new concept and do customization with prompt engineering, fine-tuning, and Retrieval Augmented Generation (RAG). RAG can be built with the from-scratch embedding model and vector store I implemented earlier. However, for learning, this time I will use LangChain framework which is more flexible and scalable. 

    For this App, following models are used. VectorStore - FAISS. Embedding - "BAAI/bge-base-en-v1.5". LLM can select mistral or llama. Both query with RAG and without RAG are performed together, and the results are displayed side by side to showcase the great improvement achieved with RAG. 

    Try it at Huggingface Space [here](https://huggingface.co/spaces/wb-droid/RetrievalAugmentedGeneration).
    <br><img src="./RAG/RAG_example.jpg" height="300">

7. <b>`Telegram Chat Bot with Local LLM/application`</b>

    Previous developed LLM generation/chat apps are web based. A chat bot in phone app such as Telegram is more accessible and has more options to customize. There are two ways to build such chat bot as shown below. Local LLM/application is easier to deploy. And developer has full control. It's also more secure. On the other hand, online LLM/application is more flexible to use multiple LLMs, and does not require local resource/maintenance. This project will use the first option: Local LLM/application. Next project will use the second option.
    <br><img src="./TelegramBot/architecture.jpg" height="240">
    
    Local LLM can be setup similar to my previous projects. But this time, I will use [text-generation-webui](https://github.com/oobabooga/text-generation-webui) for its maturity and features. Use the command below to enable API endpoint, and load a quantized Mistral 7B model.

    `start_wsl.bat --api --model TheBloke_Mistral-7B-Instruct-v0.1-GPTQ`
    
    Local chat application can be developed with Telegram API library, such as [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot). For this project, I will use [chatgpt-mirai-qq-bot](https://github.com/lss233/chatgpt-mirai-qq-bot) for it's easy to customize and full of features. With this [config.cfg](./TelegramBot/chatgpt-mirai-qq-bot/config.cfg), and update its `bot_token` with your own Telegram Bot Token (refer to https://core.telegram.org/bots). Then, run the command below.

    `docker run --name mirai-chatgpt-bot -v ./config.cfg:/app/config.cfg --network host lss233/chatgpt-mirai-qq-bot:browser-version`

    Now talk to the Mistral AI with Telegram like you talk to another person.
    <br><img src="./TelegramBot/screenshot.jpg" height="240">


8. <b>`Telegram Chat Bot with Online LLM/Webhook`</b>

    The previous Telegram Bot needs local resources to run local LLM/application. To save my resources, I also try to build the 2nd option, "Online LLM/Webhook", using only free online resource.

    a) Attempt on Huggingface is only partial successful.
    
    The Telegram webhook webserver is setup on Huggingface here: https://wb-droid-telegramwebhook.hf.space/webhook. After it's running, use the following to register this webhook with Telegram:
    https://api.telegram.org/bot{YOUR_TOKEN}/setWebhook?url={YOUR_WEBHOOK_ENDPOINT}. This webhooks works and can receive Telegram messages. But it cannot sent reply message out because Huggingface does not support connection to api.telegram.org and fails for "No address associated with hostname".

    b) Cloudflare worker works successfully.

    To workaroud the limitation in previous Huggingface webhook, I build another Telegram webhook as Cloudflare worker. This webhook will also call online Mistral LLM endpoint for text generation/chat response. The webhook script can be found [here](./TelegramBot_webhook/worker.js). After register this webhook with Telegram, I can chat with Telegram AI bot without using any local resources, and for free. 
    <br><img src="./TelegramBot_webhook/webhook_chat.jpg" height="240">

9. <b>`MyMarioAI with Deep Reinforcement Learning`</b>

    AI can play games better than human. There are various Mario AI implementations, such as [PPO AI](https://github.com/uvipen/Super-mario-bros-PPO-pytorch) and [DDQN AI](https://github.com/yfeng997/MadMario). I like MadMario's elegant design and implementation, using DDQN (Double Deep Q Network). But when I try to train it, the training is not progressing very well. This diagrom shows that total rewards barely improved after 8,000 episodes of training. 
    <img src="./MyMarioAI/Slow_training_of_madmario.jpg" height="120"> 
    The author suggested 40,000 episodes of training loops to properly train the AI. 
    
    I made some adjustments: More balanced exploration and exploitation. Reduced burn-in to start training faster. Enhanced learning for Mario death cases. Weighted learning experience to focus more on newer actions/states. Then the training is faster. <img src="./MyMarioAI/Faster_training_of_MyMarioAI.jpg" height="120"> 

    My training result: <img src="./MyMarioAI/movie_full_run.gif" height="120">
    
    Training and playback scripts can be found [here](./MyMarioAI/)

    A Huggingface app is built to demo the Mario AI in action. Try it [here](https://huggingface.co/spaces/wb-droid/MyMarioAI2). 
    <img src="./MyMarioAI/MarioAI_app.jpg" height="240"> 

10. <b>`Variational Autoencoder`</b>

    VAE (Variational Autoencoder) is one of the 3 component models of stable diffusion. After defusion generates new image at latent space, VAE is used to decode the latent image representation to restore the image into original space. 

    VAE has a decoder (down sampler) and encoder (up sampler). I implemented a simple VAE for FashionMNIST dataset. The result below (original image and generated image are side-by-side) shows that the VAE can capture the main features of the original images, and use it to restore a image close to original image. 
    <img src="./MyVAE/vae_generated_images.jpg" height="240"> 

    The model, training and inference code can be found [here](./MyVAE/vae.ipynb).

    A Huggingface app is also built to showcase the VAE. Try it [here](https://huggingface.co/spaces/wb-droid/Variational_Autoencoder). 
    <img src="./MyVAE/vae_app.jpg" height="240">     

11. <b>`UNet with DDPM`</b>

    UNet with DDPM (Denoising Diffusion Probilistic Model) is another one of the 3 component models of stable diffusion. UNet is trained with noisified images to predict the noise. Then the trained UNet is used for image generation. It goes through 1000 time steps of denoising process to generate a new image.

    I implemented a simple UNet by UNet2DModel. Trained it with FashionMNIST dataset. I tried DDPMScheduler but could not make it generate correct image. So I used fastai's DDPM sampler implementation instead. The result below shows the DDPM denoising process that removes noises step by step to generate the final image.

    <img src="./UNet_DPPM/denoising_process.jpg" height="240"> 

    The model, training and inference code can be found [here](./UNet_DPPM/MyUNet2_complete.ipynb).

    A Huggingface app is also built to showcase the diffusion process. Try it [here](https://huggingface.co/spaces/wb-droid/UNet_DPPM). 
    <img src="./UNet_DPPM/UNet_DPPM_app.jpg" height="240">     

12. <b>`CLIP model`</b>

    CLIP (Constrasitive Language-Image Pretraining) is the 3rd of the 3 component models of stable diffusion. CLIP is used to align text embedding and image embedding at latent space. CLIP can perform zero shot image classification/image captioning. It's also used to generate text/image embedding that can be used to guide image generation in diffusion. 

    I implemented a simple CLIP model. The image encoder reuses the VAE encoder implemented earlier. The text encoder is a label class embedding. Trained it with FashionMNIST dataset. The zero-shot image classification achived >90% accuracy.

    The model, training and inference code can be found [here](./MyCLIP/MyCLIP.ipynb).

    A Huggingface app is built to showcase the CLIP model. Try it [here](https://huggingface.co/spaces/wb-droid/CLIP_Model). 
    <img src="./MyCLIP/CLIP_app.jpg" height="240">  

13. <b>`Conditional Diffusion with DDPM/DDIM`</b>

    The previous unconditional diffusion randomaly generates sample from the UNet model's learned distribution. A more useful thing to do is to generate a targeted image that has certain properties or concepts. Conditional diffusion can generated this kind of image, guided by the conditional embedding what is an additonal input to the model.

    I implemented such a model with UNet2DConditionModel. Trained it with FashionMNIST dataset. The conditional embedding is simply tokenizing the image label (as label index), then go through an embedding layer, so that the model can generate certain type of images dictated by the input. 

    DDIM (Denoising Diffusion Implicit Model) is more efficient way to generate images faster than DDPM.  Both DDPM and DDIM generation of "T-Shirt" results are shown below.

    <img src="./ConditionalDiffusion/DDPM_generated.jpg" height="240"> 

    <img src="./ConditionalDiffusion/DDIM_generated.jpg" height="240"> 

    The model, training and inference code can be found [here](./ConditionalDiffusion/ConditionalDiffusion.ipynb).

    Another interesting usage of this model is to conditon on multiple features, to generate images that has multiple properties. For example, a "Bag" that looks a bit like "T-Shirt" can be generated with [this](./ConditionalDiffusion/ConditionalDiffusion_multi_features.ipynb), as shown below.
    <img src="./ConditionalDiffusion/bag_t_shirt.jpg" height="240"> 

    A Huggingface app is built to showcase the conditional diffusion model. Try it [here](https://huggingface.co/spaces/wb-droid/Conditional_Diffusion). 
    <img src="./ConditionalDiffusion/app.jpg" height="240">  


## Pretrained Models Evaluation/Fine-tuning

1. <b>`Pet classifier`</b>

    This is a fine-tuned resnet that can do simple image classification (cat or dog) task.
    
    Try it at Huggingface Spaces: [A pet classifier](https://huggingface.co/spaces/wb-droid/pet_classifier)

    References:

    The original model definition and training can be found in fastai's ["Saving a Cats v Dogs Model"](https://github.com/fastai/course22/blob/master/02-saving-a-basic-fastai-model.ipynb).<br>
    The method to build a gradio web app on Huggingface Spaces is nicely described in Tanishq's ["Gradio + HuggingFace Spaces: A Tutorial"](https://www.tanishq.ai/blog/posts/2021-11-16-gradio-huggingface.html).

2. <b>`Stable Diffusion Text Inversion`</b>

    Stable Diffusion can be fine-tuned to add new concept/style. This is an interesting scheme. 
    I trained Stable Diffusion to add a new token `<monkey-toy>` that represents my son's monkey toy: 
    
    <img src="Text_inversion/my_concept/0.jpg" width="200">
        
    After that, prompt of `"a <monkey-toy> in space suit"` can generate this: 
    
    <img src="Text_inversion/generated/003.png" width="200">

    Data, model, training and inference details can be found in this [Jupyter notebook](Text_inversion/sd_textual_inversion_training_mk.ipynb). 

    References:

    This model and training code is mainly from [Huggingface's notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb).<br>
    Additonal info can be found in [huggingface](https://huggingface.co/docs/diffusers/en/using-diffusers/loading_adapters#textual-inversion).

3. <b>`Image Captioning`</b>

    Image captioning is more flexible than classification, and privides more infomation on the objects in the image such as positional data. COCA Model claimed to have have the capability "to directly applicable to image captioning tasks as an encoder-decoder model". I tested the "monkey toy" image  
    <img src="Text_inversion/my_concept/0.jpg" width="32"> with OpenCLIP and COCA. The generated caption is `"a stuffed monkey with a crown on it 's head"`. Pretty good.

    Model and inference details can be found in this [Jupyter notebook](Image_caption/openCLIP_COCA_image_caption.ipynb). 

    References:

    The code is from this [OpenCLIP's notebook](https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_coca.ipynb). <br>
    CoCa paper is [here](https://arxiv.org/abs/2205.01917).<br>
    OpenCLIP repo is [here](https://github.com/mlfoundations/open_clip).

4. <b>`Object Detection`</b>

    Object Detection is can be used to detect the objects and object locations in image from text query. I tested the OWL-ViT model with "monkey toy" image. The detected objects and locations are below.
    <br><img src="Object_detection/0_with_objects_detected.jpg" width="200">

    Model and inference details can be found in this [Jupyter notebook](Object_detection/object_detection.ipynb). 

    References:

    The inference code is from this [notebook](https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/zero_shot_object_detection.ipynb). <br>
    OWL-ViT repo is [here](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit).

5. <b>`Model Quantization and CPU deployment`</b>

    Language Models are growing bigger and more capable. My GPU with 6G VRAM cannot run any serious model. Quantization is a good way to reduce model inference cost. 
    The ChatGLM3 model is 12G. After quantization with [chatglm.cpp](https://github.com/li-plus/chatglm.cpp), its size is reduced to 3.4G. 

    a. The quantized model runs slower on CPU than GPU, but it's reasonably acceptable on my laptop, generating at 10 tokens/s. However, when running on the free huggingface space CPU, it's crawling like a snail. If you have patience, you can try it [here](https://huggingface.co/spaces/wb-droid/ChatGLM3_Quantized).

    b. To improve this further, GPU + CPU scheme can make use of both CPU and GPU to provide a good ballance between speed and memory. First, make [this change to enable GPU/CPU mix](https://github.com/li-plus/chatglm.cpp/issues/218). Then, make [this change if you encouter same issue](https://github.com/li-plus/chatglm.cpp/issues/140). Also need to install [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Then, build and run with below to enjoy much improved speed:
    
    ```
    docker build . --network=host -t chatglm.cpp-cuda \
    --build-arg BASE_IMAGE=nvidia/cuda:12.2.0-devel-ubuntu20.04 \
    --build-arg CMAKE_ARGS="-DGGML_CUBLAS=ON"
    
    docker run -it --rm --gpus all -v $PWD:/chatglm.cpp/models chatglm.cpp-cuda ./build/bin/main -m models/chatglm-ggml.bin -i
    ```

    To enable python binding, run the following before load and run the model as usual. Note Gradio in docker need to set server name to expose port -- `demo.queue().launch(share=False, server_name="0.0.0.0")`.

    ```
    docker run -it --rm --gpus all -v $PWD:/chatglm.cpp/models chatglm.cpp-cuda /bin/bash
    docker run -p 7860:7860 -it --rm --gpus all -v $PWD:/chatglm.cpp/models chatglm.cpp-cuda /bin/bash
    python3 -m pip install torch tabulate tqdm transformers accelerate sentencepiece gradio
    
    python3.8 cli_demo.py
    python3.8 web_demo.py    
    ```

6. <b>`Image Captioning with LLaVA`</b>

    [LLaVA (Large Language and Vision Assistant)](https://github.com/haotian-liu/LLaVA) is good at multi-modal chat.   
    <img src="Text_inversion/my_concept/0.jpg" width="32"><br>
    I tested the same "monkey toy" image above. The following query 
    
    ```
    image, output = caption_image(f'../0.jpg', 'Describe the image.') 
    print(output)
    ``` 
    generates 

    ```
    The image features a small stuffed monkey toy with a crown on its head, sitting on a white surface. The monkey toy has a big smile on its face, making it appear cheerful and friendly. The toy is positioned in the center of the image, capturing attention and adding a playful touch to the scene.
    ``` 
    
    Very impressive.

6. <b>`Pil-3 multimodal model`</b>

    Phi-3-vision-128k-instruct is a lightweight, general purpose AI that works with both visual and text. It can do chat like other LLM. And it can do image captioning, OCR and other tasks. 
    With the same <img src="Text_inversion/my_concept/0.jpg" width="32">, question "What is shown in this image?" generates "The image shows a plush toy resembling a monkey with a yellow headband."
    
    Details of the evaluations can be found [here](./Phi-3/Phi3.ipynb)
