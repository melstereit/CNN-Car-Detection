# Transfer Learning 
1. Add fully connected layers to inception model and trained 2 epochs with all other layers freezed
2. Unfreezed all the top layers (173 and above --> https://www.kaggle.com/abnera/transfer-learning-keras-xception-cnn),
and trained 2 epochs 

--> 1 percent top 1 accuracy 

3. Trained for another 4 epochs 

--> 2 percent top 1 accuracy

4. Trained for another 6 epochs 
17 hours runtime 
--> 22 percept top 1 accuracy

5. Trained for another 6 epochs 
40 per val. acc, 50 train acc --> Small overfitting? 

6. Trained for another 6 epochs 
50 per val. acc, 72 train acc --> Overfitting :/


--> Model files stored here: https://github.com/marvpaul/CarDetectionModels