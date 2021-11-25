# Fall (Accident) Detection - A combined Approach using CNN and Computer Vision

In CV Fall Detection to detect if person had a fall or not is detected using Computer Vision. MediaPipe Library by Google has pretrained data for keypoints at certain important joints body, program AITrainer.py calculates angle between Shoulder, Hip and Knee. If it crosses a Threshold then a counter adds it as fall. 
Different types of falls: 
    fall while sleeping
    fall while sitting 
    fall while walking or standing
    fall from standing on support tools such as ladder or stairs
If person’s fall is detected and notified to near-dear ones or caretaker, suitable measures can be taken to prevent from increase in damage 

The solution to this is in following 2 videos. Video 1 calculates fall for certain values and Video 2 calculates fall for certain other values.

Proposed Model:

![Proposed Model](https://user-images.githubusercontent.com/67673406/143478934-ecd58591-9b5f-4a3f-baa9-0e7aafbcca1e.JPG)

https://user-images.githubusercontent.com/81099796/143458634-cff1d363-3b9d-4540-bc98-0d3aa72c25af.mp4


https://user-images.githubusercontent.com/81099796/143458557-98c69bd2-e7df-4455-b446-d71711d75f31.mp4

Accuracy and Loss:

![Accuracy and Loss](https://user-images.githubusercontent.com/67673406/143479086-08d76c83-5719-4cb9-8acd-a2cca1adf7fa.JPG)

# This project is yet to worked on integrating with a real time Activity Tracker that records person's Accelerometer and GyroScope. A sudden fall will give a different graph of Accelerometer and GyroScope. By Uniting these two outputs we can get a more Accurate prediction that a person has fallen or not.
