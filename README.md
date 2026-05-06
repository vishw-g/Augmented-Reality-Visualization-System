# 🚀 Augmented-Reality-Visualization-System ✨

You may (or may not) have heard of or seen the augmented reality Invizimals video game or the Topps 3D baseball cards. The main idea is to render in the screen of a tablet, PC or smartphone a 3D model of a specific figure on top of a card according to the position and orientation of the card. 

---

## 🧭 Where do we start?

Looking at the project as a whole may make it seem more difficult than it really is. Luckily for us, we will be able to divide it into smaller parts that, when combined one on top of another, will allow us to have our augmented reality application working. The question now is, which are these smaller chunks that we need? 

To achieve this we first have to be able to identify the flat surface of reference in an image or video frame. Once identified, we can easily determine the transformation from the reference surface image (2D) to the target image (2D). This transformation is called homography. However, if what we want is to project a 3D model placed on top of the reference surface to the target image we need to extend the previous transformation to handle cases were the height of the point to project in the reference surface coordinate system is different than zero. This can be achieved with a bit of algebra. Finally, we should apply this transformation to our 3D model and draw it on the screen. Bearing the previous points in mind our project can be divided into:

          1.  🔍 Recognize the reference flat surface.
          2.  📐 Estimate the homography.
          3.  🔄 Derive from the homography the transformation from the reference surface coordinate system to the target image coordinate system.
          4.  🎨 Project our 3D model in the image (pixel space) and draw it.

## 📊 Figure 1: Overview of the whole process that brings to life our augmented reality application.
![dum](img/image1.png)

The main tools we will use are Python and OpenCV because they are both open source, easy to set up and use and it is fast to build prototypes with them. For the needed algebra bit I will be using numpy.

---

## 🎯 Recognizing the target surface

From the many possible techniques that exist to perform object recognition I decided to tackle the problem with a feature based recognition method. This kind of methods, without going into much detail, consist in three main steps: feature detection or extraction, feature description and feature matching.

---

## 🔎 Feature extraction

Roughly speaking, this step consists in first looking in both the reference and target images for features that stand out and, in some way, describe part the object to be recognized. This features can be later used to find the reference object in the target image. We will assume we have found the object when a certain number of positive feature matches are found between the target and reference images. For this to work it is important to have a reference image where the only thing seen is the object (or surface, in this case) to be found.  We don’t want to detect features that are not part of the surface. And, although we will deal with this later, we will use the dimensions of the reference image when estimating the pose of the surface in a scene.

For a region or point of an image to be labeled as feature it should fulfill two important properties: first of all, it should present some uniqueness at least locally. Good examples of this could be corners or edges. Secondly, since we don’t know beforehand which will be, for example, the orientation, scale or brightness conditions of this same object in the image where we want to recognize it a feature should, ideally, be invariant to transformations; i.e, invariant against scale, rotation or brightness changes. As a rule of thumb, the more invariant the better.

## 📷 Figure 2: On the left, features extracted from the model of the surface I will be using. On the right, features extracted from a sample scene. Note how corners have been detected as interest points in the rightmost image.
![dum](img/image2.png)

---

## 🧾 Feature description

Once features have been found we should find a suitable representation of the information they provide. This will allow us to look for them in other images and also to obtain a measure of how similar two detected features are when being compared. This is were descriptors roll in.  A descriptor provides a representation of the information given by a feature and its surroundings. Once the descriptors have been computed the object to be recognized can then be abstracted to a feature vector,  which is a vector that contains the descriptors of the keypoints found in the image with the reference object.

This is for sure a nice idea, but how can it actually be done? There are many algorithms that extract image features and compute its descriptors and, since I won’t go into much more detail (a whole post could be devoted only to this) if you are interested in knowing more take a look at SIFT, SURF, or Harris. The one we will be using was developed at the OpenCV Lab and it is called ORB (Oriented FAST and Rotated BRIEF). The shape and values of the descriptor depend on the algorithm used and, in our case,  the descriptors obtained will be binary strings.

With OpenCV, extracting features and its descriptors via the ORB detector is as easy as:

```python
          img = cv2.imread('scene.jpg',0)
          
          # Initiate ORB detector
          orb = cv2.ORB_create()
          
          # find the keypoints with ORB
          kp = orb.detect(img, None)
          
          # compute the descriptors with ORB
          kp, des = orb.compute(img, kp)
          
          # draw only keypoints location,not size and orientation
          img2 = cv2.drawKeypoints(img, kp, img, color=(0,255,0), flags=0)
          cv2.imshow('keypoints',img2)
          cv2.waitKey(0)
🔗 Feature matching
Once we have found the features of both the object and the scene were the object is to be found and computed its descriptors it is time to look for matches between them. The simplest way of doing this is to take the descriptor of each feature in the first set, compute the distance to all the descriptors in the second set and return the closest one as the best match (I should state here that it is important to choose a way of measuring distances suitable with the descriptors being used. Since our descriptors will be binary strings we will use Hamming distance). This is a brute force approach, and more sophisticated methods exist.

For example, and this is what we will be also using, we could check that the match found as explained before is also the best match when computing matches the other way around, from features in the second set to features in the first set. This means that both features match each other. Once the matching has finished in both directions we will take as valid matches only the ones that fulfilled the previous condition. Figure 4 presents the best 15 matches found using this method.

Another option to reduce the number of false positives would be to check if the distance to the second to best match is below a certain threshold. If it is, then the match is considered valid.

📌 Figure 3: Closest 15 brute force matches found between the reference surface and the scene
https://img/image3.png

Finally, after matches have been found, we should define some criteria to decide if the object has been found or not. For this I defined a threshold on the minimum number of matches that should be found. If the number of matches is above the threshold, then we assume the object has been found. Otherwise we consider that there isn’t enough evidence to say that the recognition was successful.

🧪 With OpenCV all this recognition process can be done in a few lines of code:
python
          MIN_MATCHES = 15
          cap = cv2.imread('scene.jpg', 0)    
          model = cv2.imread('model.jpg', 0)
          # ORB keypoint detector
          orb = cv2.ORB_create()              
          # create brute force  matcher object
          bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
          # Compute model keypoints and its descriptors
          kp_model, des_model = orb.detectAndCompute(model, None)  
          # Compute scene keypoints and its descriptors
          kp_frame, des_frame = orb.detectAndCompute(cap, None)
          # Match frame descriptors with model descriptors
          matches = bf.match(des_model, des_frame)
          # Sort them in the order of their distance
          matches = sorted(matches, key=lambda x: x.distance)
          
          if len(matches) > MIN_MATCHES:
              # draw first 15 matches.
              cap = cv2.drawMatches(model, kp_model, cap, kp_frame,
                                     matches[:MIN_MATCHES], 0, flags=2)
              # show result
              cv2.imshow('frame', cap)
              cv2.waitKey(0)
          else:
              print "Not enough matches have been found - %d/%d" % (len(matches),MIN_MATCHES)
On a final note and before stepping into the next step of the process I must point out that, since we want a real time application, it would have been better to implement a tracking technique and not just plain recognition. This is due to the fact that object recognition will be performed in each frame independently without taking into account previous frames that could add valuable information about the location of the reference object. Another thing to take into account is that, the easier to found the reference surface the more robust detection will be. In this particular sense, the reference surface I’m using might not be the best option, but it helps to understand the process.

🧮 Homography estimation
Once we have identified the reference surface in the current frame and have a set of valid matches we can proceed to estimate the homography between both images. As explained before, we want to find the transformation that maps points from the surface plane to the image plane (see Figure 4). This transformation will have to be updated each new frame we process.

🖼️ Fgure 4: Homography between a plane and an image. Source: F. Moreno.
https://img/image4.png

How can we find such a transformation? Since we have already found a set of matches between both images we can certainly find directly by any of the existing methods (I advance we will be using RANSAC) an homogeneous transformation that performs the mapping, but let’s get some insight into what we are doing here (see Figure 5). You can skip the following part (and continue reading after Figure 9) if desired, since I will only explain the reasoning behind the transformation we are going to estimate.

What we have is an object (a plane in this case) with known coordinates in the, let’s say, World coordinate system and we take a picture of it with a camera located at a certain position and orientation with respect to the World coordinate system. We will assume the camera works following the pinhole model, which roughly means that the rays passing through a 3D point p and the corresponding 2D point u intersect at c, the camera center. A good resource if you are interested in knowing more about the pinhole model can be found here.

📐 Figure 5: Image formation assuming a camera pinhole model. Source: F. Moreno.
https://img/image5.png

Although not entirely true, the pinhole model assumption eases our calculations and works well enough for our purposes. The u, v coordinates (coordinates in the image plane) of a point p expressed in the Camera coordinate system if we assume a pinhole camera can be computed as (the derivation of the equation is left as an exercise to the reader):

📏 Figure 6: Image formation assuming a pinhole camera model. Source: F. Moreno.
https://img/image6.png

Where the focal length is the distance from the pinhole to the image plane, the projection of the optical center is the position of the optical center in the image plane and k is a scaling factor. The previous equation then tells us how the image is formed. However, as stated before, we know the coordinates of the point p in the World coordinate system and not in the Camera coordinate system, so we have to add another transformation that maps points from the World coordinate system to the Camera coordinate system. The transformation that tells us the coordinates in the image plane of a point p in the World coordinate system is then:

🔢 Figure 7: Computation of the projection matrix. Source: F. Moreno.
https://img/image7.png

Luckily for us, since the points in the reference surface plane do always have its z coordinate equal to 0 (see Figure 4) we can simplify the transformation that we found above. It can be easily seen that the product of the z coordinate and the third column of the projection matrix will always be 0 so we can drop this column and the z coordinate from the previous equation. By renaming the calibration matrix as A and taking into account that the external calibration matrix is an homogeneous transformation:

✂️ Figure 8: Simplification of the projection matrix. Source: F. Moreno.
https://img/image8.png

From Figure 8 we can conclude that the homography between the reference surface and the image plane, which is the matrix we will estimate from the previous matches we found is:

🧩 Figure 9: Homography between the reference surface plane and the target image plane. Source: F. Moreno.
https://img/image9.png

There are several methods that allow us to estimate the values of the homography matrix, and you maight be familiar with some of them. The one we will be using is RANdom SAmple Consensus (RANSAC). RANSAC is an iterative algorithm used for model fitting in the presence of a large number of outliers, and Figure 11 ilustrates the main outline of the process. Since we cannot guarantee that all the matches we have found are actually valid matches we have to consider that there might be some false matches (which will be our outliers) and, hence, we have to use an estimation method that is robust against outliers. Figure 10 illustrates the problems we could have when estimating the homography if we considered that there were no outliers.

⚠️ Figure 10: Homography estimation in the presence of outliers. Source: F. Moreno.
https://img/image10.png

🔁 Figure 11: RANSAC algorithm outline. Source: F. Moreno.
https://img/image11.png

As a demonstration of how RANSAC works and to make things clearer, assume we had the following set of points for which we wanted to fit a line using RANSAC:

📍 Figure 12: Initial set of points. Source: F. Moreno
https://img/image12.png

From the general outline presented in Figure 12 we can derive the specific process to fit a line using RANSAC (Figure 13).

📐 Figure 13: RANSAC algorithm to fit a line to a set of points. Source: F. Moreno.
https://img/image13.png

A possible outcome of running the algorithm presented above can be seen in Figure 14. Note that the first 3 steps of the algorithm are only shown for the first iteration (indicated by the bottom right number), and from that on only the scoring step is shown.

📊 Figure 14: Using RANSAC to fit a line to a set of points. Source: F. Moreno.
https://img/image14.png

Now back to our use case, homography estimation. For homography estimation the algorithm is presented in Figure 15. Since it is mainly math, I won’t go into details on why 4 matches are needed or on how to estimate H. However, if you want to know why and how it’s done, this is a good explanation of it.

📈 Figure 15: RANSAC for homography estimation. Source: F. Moreno.
https://img/image15.png

Before seeing how OpenCV can handle this for us we should discuss one final aspect of the algorithm, which is what does it mean that a match is consistent with H. What this mainly means is that if after estimating an homography we project into the target image the matches that were not used to estimate it then the projected points from the reference surface should be close to its matches in the target image. How close they should be to be considered consistent is up to you.

I know it has been tough to reach this point, but thankfully there is a reward. In OpenCV estimating the homography with RANSAC is as easy as:

python
          # assuming matches stores the matches found and 
          # returned by bf.match(des_model, des_frame)
          # differenciate between source points and destination points
          src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
          dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
          # compute Homography
          M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
Where 5.0 is the threshold distance to determine if a match is consistent with the estimated homography. If after estimating the homography we project the four corners of the reference surface on the target image and connect them with a line we should expect the resulting lines to enclose the reference surface in the target image. We can do this by:

python
          # Draw a rectangle that marks the found model in the frame
          h, w = model.shape
          pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
          # project corners into frame
          dst = cv2.perspectiveTransform(pts, M)  
          # connect them with lines
          img2 = cv2.polylines(img_rgb, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
          cv2.imshow('frame', cap)
          cv2.waitKey(0)
which results in:

🟦 Figure 16: Projected corners of the reference surface with the estimated homography.
https://img/image16.png

I think this is enough for today. On the next post we will see how to extend the homography we already estimated to project not only points in the reference surface plane but any 3D point from the reference surface coordinate system to the target image. We will then use this method to compute in real time, for each video frame, the specific projection matrix and then project in a video stream a 3D model of our choice from an .obj file. now go and sleep.

🧠 Augmented-Reality-Visualization-System (part 2)
So you finally woke up, We left the project in a state where we were able to estimate the homography between our reference surface and a frame that contained that same surface in an arbitrary position. To do so we were using RANSAC. Well, how do we keep building our augmented reality application from there?

🧭 Pose estimation from a plane
What we should achieve to project our 3D models in the frame is, as we have already said, to extend our homography matrix. We have to be able to project not only points contained in the reference surface plane (z-coordinate is 0), which is what we can do now, but any point in the reference space (with a z-coordinate different than 0).

If we go back to the first paragraphs of the section Homography Estimation, on part 1, we reached the conclusion that the 3×3 homography matrix was the product of the camera calibration matrix (A) by the external calibration matrix – which is an homogeneous transformation-. We dropped the third column (R3) of the homogeneous transformation because the z-coordinate of all the points we wanted to map was 0 (since all of them were contained in the reference surface plane). Figure 18 shows again the last steps of how we obtained the final expression of the homography matrix.

📐 Figure 17: Derivation of the homography matrix. Source: F.Moreno
https://img/image17.png

This meant that we were left with the equation presented in Figure 18.

📏 Figure 18: Components of the homography matrix. Source: F. Moreno
https://img/image18.png

However, we now want to project points whose z-coordinate is different than 0, which is what will allow us to project 3D models. To do so we should find a way to compute, from what we know, the value of R3. Why? Because once z is different than 0 we can no longer drop R3 from the transformation (see Figure 17) since it is needed to map the z-value of the point we want to project. The problem of extending our transformation from 2D to 3D will be solved then when we find a way to obtain the value of R3 (see Figure 18, again). But, how can we get the value of R3?

We have already estimated the homography (H) , so its value is known. Furthermore, either by looking up the camera parameters or with a bit of common sense, we can easily know or make an educated guess of the values of the camera calibration matrix (A). Remember that the camera calibration matrix was:

🎥 Figure 19: Camera calibration matrix. Source: F. Moreno
https://img/image19.png

Here you can find a nice article (part 3 of a recommended series) that talks in detail about the camera calibration matrix and each of its values, and you can even play with them. Since all I am building is a prototype application I just made an educated guess of the values of this matrix. When it comes to the projection of the optical center, I just set u0 and v0 to be half the resolution of the frames I am capturing with OpenCV (u0=320 and v0=240). As for the focal length, this article provides some useful information on how to estimate the focal length of a webcam or a cellphone camera. I set fu and fv to the same value, and found that a focal length of 800 worked quite well for me. You may have to adjust these parameters to your actual set up or even go on calibrating your camera.

Now that we have estimates of both the homography matrix H and our camera calibration matrix A, we can easily recover R1, R2 and t by multiplying the inverse of A by H:

🔄 Figure 20: Recovering the external calibration matrix values from the estimated homography and the camera calibration matrix. Source: F.Moreno
https://img/image20.png

Were the values of G1, G2 and G3 can be regarded as:

📥 Figure 21: Extracting the values of the external calibration matrix. Source: F.Moreno
https://img/image21.png

Now, since the external calibration matrix [R1 R2 R3 t] is an homogeneous transformation that maps points amongst two different reference frames we can be sure that [R1 R2 R3] have to be orthonormal. Hence, theoretically we can compute R3 as:

➕ Figure 22: Computation of R3. Source: F. Moreno
https://img/image22.png

Unluckily for us, getting the value of R3 is not as simple as that. Since we obtained G1, G2 and G3 from estimations of A and H there is no guarantee that [R1=G1 R2=G2 R3=G1xG2] will be orthonormal. The problem is then to get a pair of vectors that are close to G1 and G2 (since G1 and G2 are estimates of the real values of R1 and R2) and that are orthonormal. Once this pair of vectors has been found (R1′ and R2′) then it will be true that R3 = R1’xR2′, so finding the value of R3 will be trivial. There are many ways in which we can find such a basis, an I will explain on of them. Its main benefit, from my point of view, is that it does not directly include any angle-related computation and that, once you get the hang of it, it is quite straightforward.

Finally and before diving into the explanation, the fact that the vectors we are looking for have to be close to G1 and G2 and not just any orthonormal basis in the same plane as G1 and G2 is important in understanding why some of the next steps are required. So make sure you understand it before moving on. I will try my best in explaining the process by which we will get this new basis but if don’t succeed in doing so do not hesitate to tell me and I will try to rephrase the explanation and make it clearer. It will be useful to have at hand Figure 23 since it provides visual information that can help in understanding the process. Note that what I am calling G1 and G2 are called R1 and R2 respectively in Figure 23. Let’s go for it!

We start with the reasonable assumption that, since G1 and G2 are estimates of the real R1 and R2 (which are orthonormal), the angle between G1 and G2 will be approximately 90 degrees (in the ideal case it will be exactly 90 degrees). Furthermore, the modulus of each of this vectors will be close to 1. From G1 and G2 we can easily compute an orthogonal basis -this meaning that the angle between the basis vectors will exactly be 90 degrees- that will be rotated approximately 45 degrees clockwise with respect to the basis formed by G1 and G2. This basis is the one formed by c=G1+G2 and d = c x p = (G1+G2) x (G1 x G2) in Figure 24. If the vectors that form this new basis (c,d) are made unit vectors and rotated 45 degrees counterclockwise (note that once the vectors have been transformed into unit vectors – v / ||v|| – rotating the basis is as easy as d’ = c / ||c|| + d / ||d|| and c’ = c / ||c|| – d / ||d||), guess what? We will have an orthogonal basis which is pretty close to our original basis (G1, G2). If we normalize this rotated basis we will finally get the pair of vectors we were looking for. You can see this whole process on Figure 23.

🔧 Figure 23: Normalization of [R1 R2 R3] to guarantee that they are orthonormal. Source: F.Moreno
https://img/image23.png

Once this basis (R1′, R2′) has been obtained it is trivial to get the value of R3 as the cross product of R1′ and R2′. This was tough, but we are all set now to finally obtain the matrix that will allow us to project 3D points into the image. This matrix will be the product of the camera calibration matrix A by [R1′ R2′ R3 t] (where t has been updated as shown in Figure 23). So, finally:
3D projection matrix = A · [R1′ R2′ R3 t]

Note that this 3D projection matrix will have to be computed for each new frame. With numpy we can, in a few lines of code, define a function that computes it for us:

python
          def projection_matrix(camera_parameters, homography):
          """From the camera calibration matrix and the estimated homographycompute the 3D projection matrix"""
          # Compute rotation along the x and y axis as well as the translation
          homography = homography * (-1)
          rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
          col_1 = rot_and_transl[:, 0]
          col_2 = rot_and_transl[:, 1]
          col_3 = rot_and_transl[:, 2]
          # normalise vectors
          l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
          rot_1 = col_1 / l
          rot_2 = col_2 / l
          translation = col_3 / l
          # compute the orthonormal basis
          c = rot_1 + rot_2
          p = np.cross(rot_1, rot_2)
          d = np.cross(c, p)
          rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
          rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
          rot_3 = np.cross(rot_1, rot_2)
          # finally, compute the 3D projection matrix from the model to the current frame
          projection = np.stack((rot_1, rot_2, rot_3, translation)).T
          return np.dot(camera_parameters, projection)
Note that the sign of the homography matrix is changed in the first line of the function. I will let you think why this is required.

As a summary, let me shortly recap our thought process to estimate the 3D matrix projection.

Derive the mathematical model of the projection (image formation). Conclude that, at this point, everything is an unknown.

Heuristically estimate the homography via keypoint matching and RANSAC. -> H is no longer unknown.

Estimate the camera calibration matrix. -> A is no longer unknown.

From the estimations of the homography and the camera calibration matrix along with the mathematical model derived in 1, compute the values of G1, G2 and t.

Find an orthonormal basis in the plane (R1′, R2′) that is similar to (G1,G2), compute R3 from it and update the value of t.

🧩 Figure 24: Thought process to recover the 3D projection matrix.
https://img/image24.png

🎮 Model projection
🦊 Figure 25: Fox projection.
https://img/image25.png

We are now reaching the final stages of the project. We already have all the required tools needed to project our 3D models. The only thing we have to do now is get some 3D figures and project them!

I am currently only using simple models in Wavefront .obj format. Why OBJ format? Because I found them easy to process and render directly with bare Python without having to make use of other libraries such as OpenGL. The problem with complex models is that the amount of processing they require is way more than what my computer can handle. Since I want my application to be real-time, this limits the complexity of the models I am able to render.

I downloaded several (low poly) 3D models format from clara.io such as this fox. Quoting Wikipedia, a .obj file is a geometry definition file format. If you download it and open the .obj file with your favorite text editor you will get an idea on how the model’s geometry is stored. And if you want to know more, Wikipedia has a nice in-detail explanation.

The code I used to load the models is based on this OBJFileLoader script that I found on Pygame’s website. I stripped out any references to OpenGL and left only the code that loads the geometry of the model. Once the model is loaded we just have to implement a function that reads this data and projects it on top of the video frame with the projection matrix we obtained in the previous section. To do so we take every point used to define the model and multiply it by the projection matrix. One this has been done, we only have to fill with color the faces of the model. The following function can be used to do so:

python
    def render(img, obj, projection, model, color=False):
        vertices = obj.vertices
        scale_matrix = np.eye(3) * 3
        h, w = model.shape
  
        for face in obj.faces:
            face_vertices = face[0]
            points = np.array([vertices[vertex - 1] for vertex in face_vertices])
            points = np.dot(points, scale_matrix)
            # render model in the middle of the reference surface. To do so,
            # model points must be displaced
            points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
            dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
            imgpts = np.int32(dst)
            if color is False:
                cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
            else:
                color = hex_to_rgb(face[-1])
                color = color[::-1] # reverse
                cv2.fillConvexPoly(img, imgpts, color)

        return img
There are two things to be highlighted from the previous function:

The scale factor: Since we don’t know the actual size of the model with respect to the rest of the frame, we may have to scale it (manually for now) so that it haves the desired size. The scale matrix allows us to resize the model.

I like the model to be rendered on the middle of the reference surface frame. However, the reference frame of the models is located at the center of the model. This means that if we project directly the points of the OBJ model in the video frame our model will be rendered on one corner of the reference surface. To locate the model in the middle of the reference surface we have to, before projecting the points on the video frame, displace the x and y coordinates of all the model points by half the width and height of the reference surface.

There is an optional color parameter than can be set to True. This is because some models also have color information that can be used to color the different faces of the model. I didn’t test it enough and setting it to True might result in some problems. It is not 100% guaranteed this feature will work.

And that’s all! go to sleep i know you haven't slept for hole night.

🏆 Results
Here you can find some images that showcase the current results. As always, there are many things that can be improved but overall we got it working quite well.

🚀 Figure 26: Done. Let’s ship the code!
https://img/image26.png

specially for Linux users, make sure that your OpenCV installation has been compiled with FFMPEG support. Otherwise, capturing video will fail. Pre-built OpenCV packages such as the ones downloaded via pip are not compiled with FFMPEG support, which means that you will have to build it manually.

As usual, you can find the code of this project on GitHub. I would have liked to polish it a bit more and add a few more functionalities, but this will have to wait. I hope the current state of the code is enough to get you started.

The code might not work directly as-is (you should change the model, tweak some parameters, etc.), but with some tinkering you can surely make it work! I’ll try to help in any issues you find along the way!

If interested, here you can find the third and final part on the series, were we will learn how to smoothen and stabilise the projection:

🧘 Augmented-Reality-Visualization-System (part 3)
I know you are not sleeping from 3 days,
This is the third part of the Augmented Reality application prototype we have been building with Python and Open CVAt the end of part 2 we were able to successfully render a 3D Model on top of a reference surface, which was our main goal.In this post we will improve the algorithm so that the projection is smoother and more stable:

Although we already have in place the building blocks required for the application to work, there is still room for improvement. For example, it is easy to see that the projected model is really shaky. One of the many things we can do to smooth the model movement is to implement a tracking system. Currently, we are running a detection algorithm to find the reference surface at each frame, and the result of this detection is what we are using to project the model in the frame.

📊 Figure 27: Current application flow.
https://img/image27.png

In the current workflow we are not taking into account the information we already have from previous frames of the estimated position and velocity of the reference surface. Since our current workflow only takes into account the current frame, if our detection algorithm computation is off on one frame we have no means to know it and correct the estimation. We just work with the data from that specific frame. A tracking system will take into account the information obtained from previous frames and combine it with the current detection to obtain a final estimation of the reference surface’s position in the current frame. This will hopefully smooth the projection and make the results more visually appealing.

🎯 Choosing a tracking system
There are many ways in which a tracking system can be implemented, but I’ve always wanted to give the Kalman Filter a go, and this seemed like a good opportunity. A part from that, if the process and measurement covariances are known, the Kalman filter is the best possible linear estimator when it comes to minimising the mean-square-error.

🧠 Kalman Filter
I won’t go into the details of how a Kalman filter works, since there are plenty of good resources online. Here, I will limit myself to the problem we are trying to solve and how a Kalman filter can be put to good use. Keeping it simple, what a Kalman Filter basically does is mix data from different sources optimally to produce the best estimation of the desired system variables or states.

📡 Figure 28: Kalman filter overview. Source: Wikipedia.
https://img/image28.png

There are several ways in which a Kalman filter can be used in our application (for example we could apply it to the homography matrix). From the many ways in which we can use it, the criteria I used to select the implemented approach is simplicity (not very scientific, I know). With a sample size of one, I reached the conclusion that the easiest and most intuitive approach -although suboptimal due to the number of computations that have to be done more than once- was to track and estimate the position and velocity of the reference surface. More specifically, the position and velocity of each of its corners. As seen on figure 30, this means we will estimate the homography and the projection matrix twice for each frame. This is not very efficient, but from my point of view – and due to the nature of the project- the resulting simplicity pays off.

With that in mind, our modified application flow will now look like the following,

🔄 Figure 29: Modified application flowchart including the Kalman filter to as the tracking system.
https://img/image29.png

where the green rectangle highlights the new pieces that come into play for the tracking. To be able to implement the Kalman filter we need to define:

The state vector (x), which will contain the variables of the system we want to estimate.

The system model (A) used to predict the evolution of the state vector.

The measurements vector (z), which will contain the measurements used to correct the estimation.

The measurement model (H), used to estimate measurement values from the current state.

We have already decided which is going to be our state vector, the position and velocity of each of the 4 corners of the reference surface. This results in a 16×1 state vector, since we have 4 corners and for each corner we want to track x, y, vx and vy.

Our system model is also easy to get. It is the set of equations that we can use to predict the evolution of the state vector. In our case, since we are tracking positions and velocities, these are simple equations of motion. To further simplify the model we will also make the assumption that from frame to frame there are no accelerations. The resulting system model is really simple. The predicted evolution of each of the states will be:

For positions:

p(t+1) = p(t) + v_p(t)*dt

For velocities:

v_p(t+1) = v_p(t)

Taking this into account it is easy to build the complete system model matrix A:

text
          [[1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
           [0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
           [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
           [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
           [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
           [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
           [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
           [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1.]
           [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
           [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
           [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
Next on the list, the measurements vector, z. This vector contains the measurements of the system that that give us some ground truth to correct the estimations on each iteration. In this case, what we can easily measure on each frame are the positions of the four corners of the reference surface. We will then have an 8×1 meaurements vector that will contain the x & y coordinates of each corner.

Last but not least, the final piece of the puzzle is to define the measurement model matrix, H. This matrix should map the state vector x into the measurements vector z so that we can compare them and see how different the estimations are from our measurements. In our case, this can be done with the following simple matrix:

text
          [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
           [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
           [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
           [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]
This are the main pieces we need to implement the Kalman filter that we will use as our tracking algorithm.

We also have to define the process and measurement noise covariances, Q and R. We are relaitvely free to choose the values of this matrices as long as they are valid covariance matrices.

With respect to Q, it is useful to understand that:

A diagonal matrix means that our state parameters are independent from each other.

The bigger the values of Q with respect to P, the less confidence we have in the model being a good predictor of the process. (Note that Q is used to increase uncertainty when updating P by being added to the result of A·P·A’).

With respect to R, on some cases when we get the measurements directly from sensors it is possible to get the values from the manufacturer, since R depends on sensor sensitivity. If not, we can use the identity matrix multiplied by a scalar that is less than 1.

A part from those considerations, some trial and error always helps in determining which values of Q and R result in a good filter performance. In my case, I have gone with the following definitions for Q and R.

python
        def get_Q(self, q=0.3, **kwargs):
            a = np.eye(8)*q**2
            b = np.zeros([8, 8])
            return np.block([[b, b], [b, a]])  # 16x16
          
        def get_R(self, r=0.6, **kwargs):
                return np.eye(8) * r**2  # 8x8
And that’s it! This is all we need to be able to implement the Kalman filter as the tracking algorithm for our reference surface and, hopefully, increase the stability of the model we are projecting in the reference surface.

⚙️ Implementation
Note: The full code of the implementation can be found above in this repository

The kalman filter mainly consists in two steps:

A prediction step where the state and covariance of the system is estimated from past data. This prediction step can be divided into:

Predict state x
Predict covariance P

python
        def predict(self, **kwargs):
            """ Prediction step """
            self.__update_models_and_noise(**kwargs)
            self.__project_state()
            self.__project_covariance()
 
        def __update_models_and_noise(self, **kwargs):
            """ If any of the models has to be updated with some data, do it here"""
            self._A = self.get_A(**kwargs)
            self._H = self.get_H(**kwargs)
            self._Q = self.get_Q(**kwargs)
            self._R = self.get_R(**kwargs)
 
        def __project_state(self):
            """ x_k = A*x_(k-1) + B*u_k"""
            self.x = np.matmul(self._A, self.x)
 
        def __project_covariance(self):
            """ P_k = A*P_(k-1)*A' + Q """
            self.P = np.matmul(
                np.matmul(self._A, self.P),
                np.transpose(self._A)
            ) + self._Q
A correction step where the state of the system is corrected from some measurements. This correction step can be divided into:

Compute gain K from predicted covariance P, measurement model H and measurement covariance R

Correct state x from measurements z and gain K (and measurement model H)

Correct covariance P from gain K (and measurement model H)

python
        def correct(self, measurements):
            """ Correction step """
            self.__compute_gain()
            self.__correct_state(measurements)
            self.__correct_covariance()
 
        def __compute_gain(self):
            """ K_k = P_k*H'*(H*P_K*H' + R)^-1 """
            self.K = np.matmul(
                np.matmul(self.P, np.transpose(self._H)),
                np.linalg.inv(
                    np.matmul(
                        self._H,
                        np.matmul(self.P, np.transpose(self._H))
                    ) + self._R
                )
            )
 
        def __correct_state(self, z):
            """ x_k = x_K + K_k*(z_k - H*x_k) """
            self.x = self.x + np.matmul(
                self.K,
                z - np.matmul(self._H, self.x)
            )
 
        def __correct_covariance(self):
            """ P_k = (I - K_k*H)*P_k """
            self.P = np.matmul(np.eye(16) - np.matmul(self.K, self._H), self.P)
These are the two main steps of the process, you can check the full class implementation in the link at the beginning of the section.

Finally, what remaind to be done to use the filter in our projection process is to use the output of the filter estimation to compute the homography matrix, as stated in the diagram shown on Figure 29.

📈 Results and Analysis
Here’s a images of the results after implementing the Kalman filter. On the right you can see the original projection, on the left, the results after implementing the tracking system. As seen, it does indeed improve the stabilisation of the projection with some charts, of the obtained results after implementing the Kalman filter. It can be seen that the estimations obtained with the filter are indeed smoother.

📊 Figure 30: Number of matches found at each frame.
https://img/image30.webp

📈 Figure 31: Non filtered (blue -cx and red – cy) and filtered (green – kcx and purple – kcy) center position estimation.
https://img/image31.webp

All in all, this has been a fun project to work on, and I hope it inspires you to build something on your own! Feel free to reach me for any doubts you might have!

And for god's sake please sleep or else you shall fall asleep while the presentation
Thanks for reading until here, see you next time!
