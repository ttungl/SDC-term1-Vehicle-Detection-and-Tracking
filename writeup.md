### SDC-term1
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
    
    Tung Thanh Le
    ttungl at gmail dot com
   
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

<!-- <img src="https://github.com/ttungl/SDC-term1-Advanced-Lane-Finding/blob/master/alf.gif" height="303" width="550"> -->


<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/add_heatmap.png">
<!-- <img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/car_HOG1.png">
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/car_HOG2.png">
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/car_HOG3.png"> -->
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/detected_rectangles.png">
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/detected_rectangles0.png">
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/draw_rectangles.png">
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/heatmap_threshold_gray.png">
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/heatmap_threshold.png">
<!-- <img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/noncar_HOG1.png">
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/noncar_HOG2.png">
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/noncar_HOG3.png"> -->
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/slide_window.png">
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/slide_window1.png">
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/slide_window2.png">
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/SVC_decision_tree.png">
<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/SVC_SupportVectorMachine.png">

I will consider the [rubric points]](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I explore the datasets of `car` and `non-car` classes as below.

<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/data_exploration.png">

Then, I use `get_hog_features()` method with `hog()` from `skimage.feature` library in `cell 7` to get the histogram of oriented gradients (HOG) features in both `car` and `non-car` classes. 

<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/car_HOG1.png">

<img width="750" src="https://github.com/ttungl/SDC-term1-Vehicle-Detection-and-Tracking/blob/master/output_images/noncar_HOG1.png">

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

I tried various combinations of parameters with training process using SVM classifier and then I obtained the prediction time of the SVM classifier as below. 


| No     |  Test Accuracy  | Training time (s)| Prediction time (s) |
| :----: | :-------------: | :----------: | :-------------: |
| 1      |            |              |                 |
| 2      |            |              |                 |
| 3      |            |              |                 |
| 4      |            |              |                 |
| 5      |            |              |                 |
| 6      |            |              |                 |
| 7      |            |              |                 |
| 8      |            |              |                 |
| 9      |            |              |                 |
| 10     |            |              |                 |
| 11     |            |              |                 |
| 12     |            |              |                 |
| 13     |            |              |                 |
| 14     |            |              |                 |
| 15     |            |              |                 |
| 16     |            |              |                 |
| 17     |            |              |                 |
| 18     |            |              |                 |
| 19     |            |              |                 |
| 20     |            |              |                 |
| 21     |            |              |                 |
| 22     |            |              |                 |
| 23     |            |              |                 |
| 24     |            |              |                 |
| 25     |            |              |                 |
| 26     |            |              |                 |
| 27     |            |              |                 |
| 28     |            |              |                 |
| 29     |            |              |                 |
| 30     |            |              |                 |
| 31     |            |              |    	           |
| 32     |            |              |                 |
| 33     |            |              |                 |
| 34     |            |              |                 |
| 35     |            |              |                 |
| 36     |            |              |                 |
| 37     |            |              |                 |
| 38     |            |              |                 |
| 39     |            |              |                 |
| 40     |            |              |                 |
| 41     |            |              |                 |
| 42     |            |              |                 |
| 43     |            |              |                 |
| 44     |            |              |                 |
| 45     |            |              |                 |
| 46     |            |              |    	           |
| 47     |            |              |                 |
| 48     |            |              |                 |
| 49     |            |              |                 |
| 50     |            |              |                 |
| 51     |            |              |                 |
| 52     |            |              |    	           |
| 53     |            |              |                 |
| 54     |            |              |                 |
| 55     |            |              |    	           |
| 56     |            |              |                 |
| 57     |            |              |                 |
| 58     |            |              |                 |
| 59     |            |              |                 |
| 60     |            |              |                 |
| 61     |            |              |    	           |
| 62     |            |              |                 |
| 63     |            |              |                 |
| 64     |            |              |                 |
| 65     |            |              |                 |
| 66     |            |              |                 |
| 67     |            |              |                 |
| 68     |            |              |                 |
| 69     |            |              |                 |
| 70     |            |              |    	           |
| 71     |            |              |                 |
| 72     |            |              |                 |






| :---------------------------------------------------------------------: |
|		   			SVM classifier (LinearSVC)				   			  |
| :---------------------------------------------------------------------: |
|  No.  | `Test Accuracy (s)` | Training time (s)  |  Prediction time (s) | 
| :---: | :-----------------: | :----------------: | :------------------: | 
|   1   |					  |	                   |                      |
|   2   |					  |	                   |                      |
|   3   |					  |	                   |                      |
|   4   |					  |	                   |                      |
|   5   |					  |	                   |                      |
|   6   |					  |	                   |                      |
|   7   |					  |	                   |                      |
|   8   |					  |	                   |                      |
|   9   |					  |	                   |                      |
|   10  |					  |	                   |                      |
|   11  |					  |	                   |                      |
|   12  |					  |	                   |                      |

|   13  |					  |	                   |                      |
|   14  |					  |	                   |                      |
|   15  |					  |	                   |                      |
|   16  |					  |	                   |                      |
|   17  |					  |	                   |                      |
|   18  |					  |	                   |                      |
|   19  |					  |	                   |                      |
|   20  |					  |	                   |                      |
|   21  |					  |	                   |                      |
|   22  |					  |	                   |                      |
|   23  |					  |	                   |                      |
|   24  |					  |	                   |                      |

|   25  |					  |	                   |                      |
|   26  |					  |	                   |                      |
|   27  |					  |	                   |                      |
|   28  |					  |	                   |                      |
|   29  |					  |	                   |                      |
|   30  |					  |	                   |                      |
|   31  |					  |	                   |                      |
|   32  |					  |	                   |                      |
|   33  |					  |	                   |                      |
|   34  |					  |	                   |                      |
|   35  |					  |	                   |                      |
|   36  |					  |	                   |                      |

|   37  |					  |	                   |                      |
|   38  |					  |	                   |                      |
|   39  |					  |	                   |                      |
|   40  |					  |	                   |                      |
|   41  |					  |	                   |                      |
|   42  |					  |	                   |                      |
|   43  |					  |	                   |                      |
|   44  |					  |	                   |                      |
|   45  |					  |	                   |                      |
|   46  |					  |	                   |                      |
|   47  |					  |	                   |                      |
|   48  |					  |	                   |                      |

|   49  |					  |	                   |                      |
|   50  |					  |	                   |                      |
|   51  |					  |	                   |                      |
|   52  |					  |	                   |                      |
|   53  |					  |	                   |                      |
|   54  |					  |	                   |                      |
|   55  |					  |	                   |                      |
|   56  |					  |	                   |                      |
|   57  |					  |	                   |                      |
|   58  |					  |	                   |                      |
|   59  |					  |	                   |                      |
|   60  |					  |	                   |                      |

|   61  |					  |	                   |                      |
|   62  |					  |	                   |                      |
|   63  |					  |	                   |                      |
|   64  |					  |	                   |                      |
|   65  |					  |	                   |                      |
|   66  |					  |	                   |                      |
|   67  |					  |	                   |                      |
|   68  |					  |	                   |                      |
|   69  |					  |	                   |                      |
|   70  |					  |	                   |                      |
|   71  |					  |	                   |                      |
|   72  |					  |	                   |                      |







####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

