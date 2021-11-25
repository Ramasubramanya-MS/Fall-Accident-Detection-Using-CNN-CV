# Fall (Accident) Detection - A combined Approach using CNN and Computer Vision

In CV Fall Detection to detect if person had a fall or not is detected using Computer Vision. MediaPipe Library by Google has pretrained data for keypoints at certain important joints body, program AITrainer.py calculates angle between Shoulder, Hip and Knee. If it crosses a Threshold then a counter adds it as fall. 

The solution to this is in following 2 videos. Video 1 calculates fall for certain values and Video 2 calculates fall for certain other values.


https://user-images.githubusercontent.com/81099796/143458634-cff1d363-3b9d-4540-bc98-0d3aa72c25af.mp4


https://user-images.githubusercontent.com/81099796/143458557-98c69bd2-e7df-4455-b446-d71711d75f31.mp4

# This project is yet to worked on integrating with a real time Activity Tracker that records person's Accelerometer and GyroScope. A sudden fall will give a different graph of Accelerometer and GyroScope. By Uniting these two outputs we can get a more Accurate prediction that a person has fallen or not.
