# COVID-19-Face-mask-detector using simple CNN 
## (1) Development environment

* Operating System : Window 10 edu
* Language : Python
* Development Tools : Pycharm Community Edition 2020.2.1
* Library : opencv, tensorflow, keras, pillow, os, ...

## (2) Implementation contents

* It can distinguish whether people are wearing mask or not.

* The results are shown in monitor.

* It can distinguish both black and white mask.

* It is designed not particular lighting environmnet but general lighting environment.

  (The model learned many different lighting environment pictures.)

* The model learned pictures with RGB scale, not gray scale.

## (3) Design methods

### ① Data collection

It seemed hard to collect photo of a face wearing a mask by taking a picture one by one or searching on the internet.

So i used facial landmark to solve the problem.

Using shape_predictor_68_landmark, it can express the part of the face with 68 numbers.

![facial_landmark](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FeXw8SC%2FbtqQwnS7TZx%2Fxvv1DIg3QxjjdcuCfqzdpk%2Fimg.png)

I choose main point number for 3, 8, 13, 29.

3 : Left chin

8 : Lower chin

13 : Right chin

29 : Center of the nose

With the chosen point, I put the mask image on photo of a face.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FRxpwI%2FbtqQJDNaGuA%2FwCmKjEVmPRE1dgDnf7KK70%2Fimg.png" width="235">

![Black_mask](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FxJLYv%2FbtqQwmUjmTD%2FPvx5MhVKQiLj6VfkhImVuk%2Fimg.png)

![image1](./Readme_images/a1.jpg)

So i tried to fit the mask on different face size of photoes.

From here, there is a problem with black mask image.

When i used the black mask image which color is (0,0,0) to train model, the model cannot distinguish the real black mask on camera.

Because on camera, there is light reflection on the black mask.

So i lowered the black transparency of the mask image.

and i useed the black mask image below.

<img src="./Readme_images/a3.jpg" width="235">

As a result, i collected the mask data.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FF6bLG%2FbtqQKvnT1bI%2FOAKNfA1JQ09WgH104dTTq1%2Fimg.png">


### ② Data Preprocessing

Also i collected face images which are not wearing mask.

But what the model really need is only the face part not the body part. 

To extract the face part, i used face recognition model.

<img src="./Readme_images/a2.jpg">


### ③ Machine learning

I designed neural network using CNN.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcUGIC8%2FbtqQDFSPj6R%2FTHS4vlGeXhtbju0T0FGjiK%2Fimg.png">
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FLq9qD%2FbtqQCzL8LVj%2FBm9AI6WePcfpBpwBmFzD31%2Fimg.png">


#### Visual graph

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb1wCtO%2FbtqQBGLusI6%2FLtFZ9TmFrcUqf1wyuchuKk%2Fimg.png">


### ④ Test

I prepared the test data which does not overlap with train data.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcKoHGV%2FbtqQBHXXPZF%2FHGER078HfSCB5jk6u7VQy0%2Fimg.png">


### ⑤ Result

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcKbjs6%2FbtqQzRNjgpq%2FTcQagcvcFoOleh3syEJjNk%2Fimg.png">


## (4) Code description

* Create_Mask_Dataset .py : Code to put the mask image into the face image

* Face_Dataset_Save .py : Code to save the only face part images in particular directory 

* Data_Deeplearning .py : Neural network code designed in CNN

* Model_Test .py : Code to test the model

* mainvideo .py : main code (if you want to test the model, you just run this code.) 

* 8LBMI2.h5 : trained model


## Reference

[Face dataset 1](https://github.com/prajnasb/observations)

[Face dataset 2](https://generated.photos/faces)

