## Description

A device running Android 5.0 (API 21) or higher is required to run the demo due
to the use of the camera2 API, although the native libraries themselves can run
on API >= 14 devices.

## 나의 환경 

Android Studio = 4.1  
Android SDK = Android 11.0(R)
Gradle = 4.0.1
SDK TOOL   
1. Android SDK Build-Tools
2. NDK(Side by side)
3. CMAKE
4. Android SDK Platform-Tools

## model

자신의 모델을 /assets에 넣으시오.  
You Should put pbfile in /assets

나의 모델은 다음과 같습니다.  
My model is tiny-yolo-4c.pb

다운로드를 하시려면 아래의 링크를 이용하십시오.  

[tiny-yolo-4c.pb](https://drive.google.com/drive/folders/1GQi7oDNKyLyl5TYsNJ48X02gYgcdNsiL?usp=sharing)



## 설명 

#### 준비물 : 삼각대 , 스마트폰 
#### Step 1   
아웃라인 설정  
Set out-line  

배드민턴 코트 라인 바깥쪽을 클릭하십시오.  
(반시계 방향)  

Click 4-point out-line  
(Counterclockwise)  

![step1](../python_detection/readme/step1.png)

#### Step 2

시작 버튼을 누르십시오.  
Click Start Button


___________
1. DectectorActivity.java  
모델 설정 및 Draw 기능 담당   
   
2. TensorFlowYoloDetector.java
물체 인식 예측 부분 
   
3. CameraActivity.java  
Camera 관련 설정  



## Milestone

1. 다방면 인아웃 판독 제공.
2. 셔틀콕 인식 성능 향상 (현재 예측 및 정확도 부분에서 기존 PC보다 성능이 떨어짐.)
3. 배드민턴 시설 예약, 물품 구매 등 커뮤니티 기능 추가 
