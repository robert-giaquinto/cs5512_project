# Introduction
Interaction with the environment is an important aspect of artificial intelligent systems that operate under uncertainty. In particular, human computer interaction provides a way for agents to be more perceptive and dynamic. One example of this sort of interaction is understanding gestures and actions. Gesture recognition for humans is so intuitive that a single example can be sufficient for learning. Whereas for computers this can be a complex classification problem. However, in recent years there have been a number of competitions devoted to solving gesture recognition given only one training example[1][2]. We seek to build on such efforts by classifying sequences of frames (containing gestures) into actions. By developing higher level of abstraction, artificial intelligent systems can intuitively understand and interact with a range of environments. With the ability to perceive human actions, computers and robots can provide more utility and more closely align longer term goals within artificial intelligence and HCI.

# Data
We plan to use data from the ChaLearn 2015 Action/Interaction Recognition competition[3]. The data consist of over 200 action instances annotated based on 11 different actions. A few examples of the actions contained in the videos include crouching, waving, walking, and hugging. In total the dataset contains over 8000 frames from RGB video. While the dataset includes labels of the action in each video, the individual gestures that make up an action are unknown.

In order to classify images we will need to apply computer vision techniques to build predictive features. Since computer vision is not the primary goal of this project, we plan to augment our data using off-the-shelf algorithms from the OpenCV[4] library to extract outlines, points of interest, and apply background elimination. The extraction of these features is a crucial first step in many object tracking and recognition tasks[5].

# Approach
Classifying actions consists of two major parts: (1) identifying low-level features for recognizing gestures and poses, and (2) using a second layer of classifiers to identify sequences of gestures and poses that make up an action. This approach follows that of Nguyen et al., which takes a multi-stage approach to gesture recognition[6]. Nguyen begins with using a hidden Markov model to recognize gestures, then using a second layer of hidden Markov models to recognize actions.

The first step, identifying gestures and poses, may prove difficult since our dataset isn't annotated with gestures or poses in each frame. We tentatively plan to use unsupervised learning methods such as Principal Component Analysis and K-Means clustering to group similar frames together and isolate important features within the frames.


[1] Isabelle Guyon, Vassilis Athitsos, Pat Jangyodsuk, and Hugo Jair Escalante. The chalearn gesture dataset (cgd 2011). Mach. Vision Appl., 25(8):1929–1951, November 2014.

[2] Isabelle Guyon, Vassilis Athitsos, Pat Jangyodsuk, Ben Hamner, and Hugo Jair Escalante. Chalearn gesture challenge: Design and first results. In CVPR Workshops, pages 1–6. IEEE, 2012.

[3] ChaLearn. Chalearn looking at people 2015 - track 2: Action recognition. https://www.codalab.org/ competitions/2241#participate.

[4] G. Bradski. Computer vision with the opencv library. Dr. Dobb’s Journal of Software Tools, 2000.

[5] Alper Yilmaz, Omar Javed, and Mubarak Shah. Object tracking: A survey. ACM Comput. Surv., 38(4),
December 2006.

[6] Nhan Nguyen-Duc-Thanh, Sungyoung Lee, and Donghan Kim. Two-stage hidden markov model in gesture recognition for human robot interaction. International Journal of Advanced Robotic Systems, 9(39), 2012.
<!---[7] Sushmita Mitra and Tinku Acharya. Gesture recognition: A survey. Systems, Man, and Cybernetics, Part C: Applications and Reviews, IEEE Transactions on, 37(3):311–324, 2007.
[8] Ruben Glatt, José C Freire Jr, and Daniel JBS Sampaio. Proposal for a deep learning architecture for activity recognition. International Journal of Engineering & Technology, 14(5), 2014.
[9] N Neverova, C Wolf, GW Taylor, and F Nebout. Multi-scale deep learning for gesture detection and localization. In ECCV Workshops, 2014.
-->
