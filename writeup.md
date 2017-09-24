### SDC-term1
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
    
    Tung Thanh Le
    ttungl at gmail dot com
   
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
	+ Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
	+ Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.

* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

<!-- <img src="https://github.com/ttungl/SDC-term1-Advanced-Lane-Finding/blob/master/alf.gif" height="303" width="550"> -->



<!-- <img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/SVC_decision_tree.png"> -->


I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I explore the datasets of `car` and `non-car` classes as below.

<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/data_exploration.png">

Then, I use `get_hog_features()` method with `hog()` from `skimage.feature` library in `cell 7` to get the histogram of oriented gradients (HOG) features in both `car` and `non-car` classes. 

<img width="650" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/car_HOG1.png">

<img width="650" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/noncar_HOG1.png">

In the images above, I obtained the HOG features using configurations `orient`= `9`, `pix_per_cell` = `8`, `cell_per_block` = `2`. Then, extracting its features using `extract_features()` as in `cell 10`. In this method, I did try the combination of HOG features, spatial, and histogram colors, but it didn't work out due to my hardware limit (macbookpro). So, I only use HOG features. 

I also tested with different configurations of `color_space`, `hog_channel`, `orient`, and `pix_per_cell`. The sizes of `car_features` and `noncar_features` are `8792` and `8968`, respectively. The table below shows the various extracted times for each configuration that I have explored.


| No     | ColorSpace | Orient       | Pixels Per Cell | HOG Channel | Extracted Time (s) |
| :----: | :--------: | :----------: | :-------------: | :---------: | --------------:|
| 1      | RGB        | 8            | 8               | 2      	 | 30.5448	       |
| 2      | RGB        | 8            | 12              | 2      	 | 22.87939	       |
| 3      | RGB        | 8            | 16              | 2      	 | 21.77754	       |
| 4      | RGB        | 11           | 8               | 2      	 | 34.036	       |
| 5      | RGB        | 11           | 12              | 2      	 | 23.02448	       |
| 6      | RGB        | 11           | 16              | 2      	 | 22.21846	       |
| 7      | RGB        | 8            | 8               | ALL         | 71.31848	       |
| 8      | RGB        | 8            | 12              | ALL         | 49.66749	       |
| 9      | RGB        | 8            | 16              | ALL         | 45.88498	       |
| 10     | RGB        | 11           | 8               | ALL         | 75.53337	       |
| 11     | RGB        | 11           | 12              | ALL         | 62.8071	       |
| 12     | RGB        | 11           | 16              | ALL         | 59.53764	       |
| 13     | HSV        | 8            | 8               | 2      	 | 31.24532	       |
| 14     | HSV        | 8            | 12              | 2      	 | 22.53092	       |
| 15     | HSV        | 8            | 16              | 2      	 | 21.72739	       |
| 16     | HSV        | 11           | 8               | 2      	 | 30.96028	       |
| 17     | HSV        | 11           | 12              | 2      	 | 23.11023	       |
| 18     | HSV        | 11           | 16              | 2      	 | 22.52908	       |
| 19     | HSV        | 8            | 8               | ALL      	 | 74.14904	       |
| 20     | HSV        | 8            | 12              | ALL      	 | 50.25714	       |
| 21     | HSV        | 8            | 16              | ALL      	 | 47.83314	       |
| 22     | HSV        | 11           | 8               | ALL      	 | 76.50115	       |
| 23     | HSV        | 11           | 12              | ALL      	 | 944.93221	   |
| 24     | HSV        | 11           | 16              | ALL      	 | 60.16057	       |
| 25     | HLS        | 8            | 8               | 2       	 | 31.15222	       |
| 26     | HLS        | 8            | 12              | 2	         | 23.32326	       |
| 27     | HLS        | 8            | 16              | 2	         | 22.1619	       |
| 28     | HLS        | 11           | 8               | 2	         | 32.06209	       |
| 29     | HLS        | 11           | 12              | 2	         | 24.07181	       |
| 30     | HLS        | 11           | 16              | 2	         | 23.87908	       |
| 31     | HLS        | 8            | 8	           | ALL	   	 | 73.75597	       |
| 32     | HLS        | 8            | 12              | ALL	     | 50.81686        |
| 33     | HLS        | 8            | 16              | ALL	     | 45.66461	       |
| 34     | HLS        | 11           |                 | ALL	     | 77.44444	       |
| 35     | HLS        | 11           | 12              | ALL	     | 50.32718	       |
| 36     | HLS        | 11           | 16              | ALL	     | 49.44808	       |
| 37     | LUV        | 8            | 8               | 2	         | 31.49909	       |
| 38     | LUV        | 8            | 12              | 2	         | 23.77771	       |
| 39     | LUV        | 8            | 16              | 2	         | 23.79319	       |
| 40     | LUV        | 11           | 8               | 2	         | 34.82909	       |
| 41     | LUV        | 11           | 12              | 2	         | 26.05469	       |
| 42     | LUV        | 11           | 16              | 2	         | 26.06044	       |
| 43     | LUV        | 8            | 8               | ALL	   	 | 75.56331	       |
| 44     | LUV        | 8            | 12              | ALL	 	 | 49.6141 	       |
| 45     | LUV        | 8            | 16              | ALL	 	 | 47.20092	       |
| 46     | LUV        | 11           | 8	           | ALL	 	 | 73.82762	       |
| 47     | LUV        | 11           | 12              | ALL	 	 | 60.64072	       |
| 48     | LUV        | 11           | 16              | ALL	 	 | 54.12637		   |
| 49     | YUV        | 8            | 8               | 2	     	 | 28.5655	       |
| 50     | YUV        | 8            | 12              | 2	    	 | 21.32812	       |
| 51     | YUV        | 8            | 16              | 2	         | 22.29023	   |
| 52     | YUV        | 11           | 8	           | 2	         | 31.1985	   |
| 53     | YUV        | 11           | 12              | 2	         | 24.73312	   |
| 54     | YUV        | 11           | 16              | 2	   	 	 | 22.23977	   |
| 55     | YUV        | 8            | 8	           | ALL	   	 | 73.98175	   |
| 56     | YUV        | 8            | 12              | ALL	   	 | 48.06844    |
| 57     | YUV        | 8            | 16              | ALL	   	 | 45.44458	   |
| 58     | YUV        | 11           | 8               | ALL	   	 | 74.28496	   |
| 59     | YUV        | 11           | 12              | ALL	   	 | 53.30763	   |
| 60     | YUV        | 11           | 16              | ALL	   	 | 46.96956	   |
| 61     | YCrCb      | 8            | 8	           | 2	     	 | 29.37965	   |
| 62     | YCrCb      | 8            | 12              | 2	     	 | 24.49854	   |
| 63     | YCrCb      | 8            | 16              | 2	     	 | 22.22448	   |
| 64     | YCrCb      | 11           | 8               | 2	     	 | 33.09982	   |
| 65     | YCrCb      | 11           | 12              | 2	     	 | 24.07404	   |
| 66     | YCrCb      | 11           | 16              | 2	    	 | 22.83741	   |
| 67     | YCrCb      | 8            | 8               | ALL	   	 | 76.33727	   |
| 68     | YCrCb      | 8            | 12              | ALL    	 | 51.09298    |
| 69     | YCrCb      | 8            | 16              | ALL	   	 | 47.65726	   |
| 70     | YCrCb      | 11           | 8	           | ALL	   	 | 80.50953	   |
| 71     | YCrCb      | 11           | 12              | ALL	   	 | 56.51481	   |
| 72     | YCrCb      | 11           | 16              | ALL	   	 | 53.57784	   |


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters with training process using SVM classifier and then I obtained the prediction time of the SVM classifier as below. I set `20` labels for prediction test.


| No     |  Test Accuracy  | Training time (s)| Prediction time (s) |
| :----: | :-------------: | :----------: | :-------------: |
| 1      |   0.96199       |   3.25232    |     0.00199     |
| 2      |   0.94707       |   1.02382    |     0.00158     |
| 3      |   0.94481         |  0.45906            |    0.00159             |
| 4      |   0.96283         |   4.0344           |     0.00146            |
| 5      |   0.95411         |    1.30958          |    0.0015             |
| 6      |   0.94763         |    0.49196          |    0.00127            |
| 7      |   0.97128         |    16.71988          |    0.00159             |
| 8      |   0.96452         |   4.23094           |     0.00181            |
| 9      |   0.96340         |    1.90431          |     0.00148            |
| 10     |   0.97606         |    21.54367          |    0.00167             |
| 11     |   0.96621         |   6.1939           |     0.00147            |
| 12     |   0.95974         |    2.81912          |    0.00154             |
| 13     |   0.96227         |    2.905          |      0.00144           |
| 14     |   0.94453         |    0.79849          |    0.00171             |
| 15     |   0.94566         |     0.46572         |    0.00153             |
| 16     |   0.96790         |    4.47107          |    0.00143             |
| 17     |   0.95579         |    0.97508          |    0.00147             |
| 18     |   0.95298         |    0.52594          |    0.00153             |
| 19     |   0.97972         |    6.04484          |    0.00156             |
| 20     |   0.98057         |    1.91117          |    0.00159             |
| 21     |   0.98001         |    0.77001          |    0.0014             |
| 22     |   0.98226         |    11.37182          |    0.00164             |
| 23     |   0.98282         |    2.65571          |     0.00245            |
| 24     |   0.97888         |     1.09589         |     0.00169            |
| 25     |   0.91779         |     6.2063         |      0.00146           |
| 26     |   0.87077         |    1.11021          |     0.00149            |
| 27     |   0.86655         |    0.56766          |     0.00142            |
| 28     |   0.91497         |     6.9495         |     0.00135            |
| 29     |   0.86796         |     1.65925         |     0.00146            |
| 30     |   0.88260         |    0.62948          |     0.00179            |
| 31     |   0.98141         |    6.51718          |    0.00156	           |
| 32     |   0.97663         |     1.9199         |     0.0015            |
| 33     |   0.97972         |     0.87492         |    0.00154             |
| 34     |   0.98141         |    9.34156          |    0.00151             |
| 35     |   0.97691         |     2.47415         |    0.00172             |
| 36     |   0.97184         |    1.26524          |    0.00161             |
| 37     |   0.91525         |    5.04221          |    0.00142             |
| 38     |   0.90568         |     1.01682         |    0.0015             |
| 39     |   0.89470         |    0.43014         |    0.00153             |
| 40     |   0.91863         |     6.03426         |    0.00173             |
| 41     |   0.91328         |     1.26704         |    0.00141             |
| 42     |   0.90822         |     0.6044         |    0.00148             |
| 43     |   0.96959         |     9.18553         |   0.00157              |
| 44     |   0.97438         |     2.21565         |   0.00181              |
| 45     |   0.96931         |     1.39191         |   0.00145              |
| 46     |   0.98141         |    13.68493          |   0.00187	           |
| 47     |   0.97466         |    2.3918          |   0.00152              |
| 48     |   0.97353         |    1.0847          |   0.00146              |
| 49     |   0.92764         |    3.80079          |   0.00149              |
| 50     |   0.91159         |     0.87636         |   0.00283              |
| 51     |   0.91835         |    0.32852          |   0.00147              |
| 52     |   0.93046         |     5.66608         |   0.00165 	           |
| 53     |   0.91497         |     1.33274         |   0.00142              |
| 54     |   0.91976         |     0.56196         |   0.00141              |
| 55     |   0.98395         |     5.16288         |   0.00154 	           |
| 56     |   0.98170         |     1.66461         |   0.0014              |
| 57     |   0.98198         |     0.81516         |   0.00148              |
| 58     |   0.98789         |     8.25317         |   0.00165              |
| 59     |   0.97888         |     2.20205         |   0.00144              |
| **60**     |   **0.98423**         |     **1.10413**         |   **0.00165**              |
| 61     |    0.93327        |     3.79404         |   0.00148 	           |
| 62     |    0.92426        |    0.90754          |   0.00182              |
| 63     |    0.92173        |   0.43846           |   0.00147              |
| 64     |    0.93834        |     6.1774         |     0.00208            |
| 65     |    0.92483        |    1.26392          |    0.00145             |
| 66     |    0.91835        |      0.54743        |    0.00188             |
| 67     |    0.97860        |     5.5966         |    0.00188             |
| 68     |    0.98170        |     1.909         |    0.00156             |
| 69     |    0.98338        |     0.94379         |    0.00145             |
| 70     |    0.98338        |     9.35322         |    0.0016	           |
| 71     |    0.97860        |     2.49796         |    0.00151             |
| 72     |    0.98338        |    1.03799          |    0.00143             |

From the results obtained as above, I observed that the parameters corresponding to the line `60` has the best result, in terms of the test accuracy `98.423`% and the training time `1.10413` seconds. The best configuration for my case is: `color_space`=`YUV`, `hog_channel`=`ALL`, `orient`= `11`, `pix_per_cell`=`16`, `cell_per_block`= `2`.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

As shown in `cell 15`, I first created a classifier using `LinearSVC()`. Then, I trained a linear SVM using `fit()` method, and obtained the test accuracy using `score()` method. I used `predict()` to obtain the predicted results. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search method is inherited from `find_cars()` method from the lesson. The method extracts the individual channel HOG features for the entire image, then the full image features are subsampled to the window size to feed to the classifier. The method performs the prediction based on HOG features for each window and then returns a number of rectangles that are predicted as detected cars.   

The image below shows the number of cars detected, it contains overlapping rectangles (brown ones) and falsely positive rectangles (green one). Note that, this result is for only one configuration of `ystart` = `400`, `ystop` = `656`, and `scale`=1.5. 

<img width="650" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/slide_window.png"> 

The images below indicate the all sliding window searches with various overlaps in both X and Y axes, with small, medium, and large window sizes as in cells 25, 26, 106, respectively. 

<img width="290" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/slide_window1.png">  <img width="290" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/slide_window2.png"> <img width="290" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/slide_window4.png"> 

Thereby, the combined results of different configurations in various window sizes is as below. The rectangles are returned from `find_cars()` method. It indicates that there are some overlaps on the detected cars and falsely positive one. Note that, I have also tried to adjust the `scale` less than `1.0` but it didn't work.

<img width="650" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/detected_rectangles.png">

Now, from the above result, I used the heatmap and threshold to eliminate the falsely positive rectangles (usually only one detection). First, `add_heat` method takes the detected rectangles from `find_cars()` and forms the heatmap image where the overlapping rectangles reside as in the left result. Then, I applied the zero out pixels below `threshold`, so it eliminates the falsely positive rectangles as shown in the middle and the right ones.

<img width="290" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/add_heatmap.png"> <img width="290" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/heatmap_threshold.png"> <img width="290" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/heatmap_threshold_gray.png"> 

Then, the thresholded heatmap image is labeled using `label` from `scipy.ndimage.measurements` to identify individual blobs in the heatmap. Finally, the (true positive) detected rectangles are drawn on the image as below.

<img width="650" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/draw_rectangles.png">


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

<img width="850" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/SVC_SupportVectorMachine.png">

The implementation performs perfect, detecting the near and far vehicles in the images without falsely positives. 

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For processing the video frames, the `pipeline_process_update()` method (cell 113) performs the same as processing an image, except that I store the detected rectangles (from `find_cars()`) of `22` previous frames to the `prev_rects` of class `Vehicle_Detect()`. I used these to feed to the heatmap, threshold (more than half of detected rectangles), and label. This helps the implementation performing very well, empirically, as the rectangles don't pop/disappear too quick.      

---
### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline performs perfectly without any issues. One problem that I have faced was the rectangles were appeared and then disappeared too quick. To overcome this, I used the `Vehicle_Detect` class to store the previous frames' detected rectangles to feed to the heatmap, threshold, and level. This helps to fix the issue. 

One technique can be used to boost the performance is using the [YOLO](https://pjreddie.com/darknet/yolo/). I will implement this when I have more time.

