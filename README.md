# GAN
## GAN과 관련된 논문을 구현
## 각 논문마다 pytorch, tensorflow로 구현하려고 노력 중

### GAN이 무엇인가?
<img src = "https://post-phinf.pstatic.net/MjAxODA4MjRfNTkg/MDAxNTM1MDcxMzI2MzU3.1-EfJtLhJXRtO5cpOBhPmY_78sHXdLKlp4_dkPjAFTQg.lIW07gOqN4zT_47N6Jik8QEv-6TShocejgoK_nBV538g.PNG/1.PNG?type=w1200">
<img src = "https://t1.daumcdn.net/cfile/tistory/9928E6375B75872D17">

- 위 그림은 GAN의 예시를 잘 나타내는 그림이다    
  - 위조 지폐범이 가짜 지폐를 생성한다  
  - 경찰이 해당 지폐가 가짜인지 진짜인지 판별한다  
  - 위의 과정이 반복 된다면 위조 지폐범은 점점 정교하게 지폐를 생성할 것이다  
  - 경찰 또한 마찬가지로 진짜 같은 가짜 지폐가 생성될수록 이를 진짜 지폐와 판별하려고 노력할 것이다  

- 위의 과정에서 위조 지폐범이 Generative model, 경찰이 Discriminator model에 해당한다
- GAN은 Generative model과 Discriminator model이 서로 적대적으로 훈련하며 성능을 높이는 방식으로 학습된다


### GAN은 어떻게 활용되는가?  
<img src = "https://post-phinf.pstatic.net/MjAxODA5MTRfMTQg/MDAxNTM2OTExMzUyNzgx.68AVr4HXMzoO5FXJfx2pVUMGD_WxoS-VpszKeuzVxUIg.gHBEL31cN2IvjSCWmq1SieXIpxq86-1lRjJvR1InKJ0g.PNG/4.PNG?type=w1200">

- 위 그림처럼 어떤 화가의 그림체를 실제 사진처럼 바꾸거나 반대로도 할 수 있다  
- 또한 얼룩말을 말로, 말을 얼룩말로 바꿀 수도 있고  
- 어떤 풍경의 여름 배경을 겨울 배경으로 바꿀 수도 있다  
- 아무튼 굉장히 다양한 분야에 쓰일 수 있다  

<img src = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ-qAX3fu25mpreT-teeFWUaA8uSbkADM-7RQ&usqp=CAU">

- 위의 첫 번째 컬럼의 이미지는 선글라스를 쓰고 있는 남자 이미지,  
- 두 번째 컬럼 이미지는 선글라스를 쓰고 있지 않은 남자 이미지,  
- 세 번째 컬럼 이미지는 선글라스를 쓰고 있지 않은 여자 이미지  
- 마지막 이미지는 선글라스를 끼고 있는 여자 이미지다  
- 즉, "선글라스를 끼고 있는 남자 - 남자 + 여자 = 선글라스를 끼고 있는 여자"로 표현할 수 있다  
- GAN을 활용한다면 이뿐만 아니라 옷, 헤어, 악세사리 등 가상 스타일링을 할 수 있다  

<img src = "https://post-phinf.pstatic.net/MjAxODA5MTRfOSAg/MDAxNTM2OTExMjkzMTUx.bnRyP_mTW_2jZnz38XGMO0a6CaXQj_KQSnE1KUidXfIg.upHzPx3nyuy5PA8YGKklru_x-3bv2wgmDEXw-iP7xy0g.PNG/2.PNG?type=w1200">

- 위 사진은 미국 전 대통령 버락 오바마다  
- 그러나 진짜 오바마가 아니라 GAN을 활용해 오바마 전 대통령이 말하는 것처럼 보이도록 한 것이다  
- 관련된 유튜브 링크는 밑에 걸어두었다    
- https://www.youtube.com/watch?v=krOXHbPcTrs&t=9s
