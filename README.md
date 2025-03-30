# RecycLENS
*Redefining Recycling, one photo at a time.* 

## Project Overview

### Front end
Used XCODE and SWIFT to create the basis for our frontend design. We used free-hand designed logos and icon for decorative and stylistic flair.

### Back end 
Coded completely in PYTHON. Used the MOBILERESNETv2 model from Tensorflow to fine tune the model on a dataset of different classes of trash. Our model had a 95% accuracy rate on test data. 

### Connecting front and back end 
Used FLASK package in PYTHON to create a back end server where the front and back end could successfully communicate. We defined functions on the front end using SWIFT to send image metadata to the back end server, where the data would then be processed and classified by our trained model. We wrote a function in the back end portion of the program to send the correct classification back so it could be displayed on our front end. 
