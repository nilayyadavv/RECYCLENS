# RecycLENS
*Redefining Recycling, one photo at a time.* 
## Project Overview
### Front end
Used XCODE and SWIFT to create the basis for our frontend design. We used free-hand designed logos and icons for decorative and stylistic flair.
### Back end 
Coded completely in PYTHON. We also used the MOBILERESNETv2 model from Tensorflow to fine-tune the model on a dataset of different trash classes. Our model had a 95% accuracy rate on test data. 
### Connecting front and back end 
Used the FLASK package in PYTHON to create a back-end server where the front and back end could successfully communicate. We defined functions on the front end using SWIFT to send image metadata to the back-end server, where the data would then be processed and classified by our trained model. We wrote a function in the back-end portion of the program to send the correct classification back so it could be displayed on our front end. 
## Running RecycLENS on your local device 
Preferably, use a Mac OS laptop so you can download the XCODE application. This will allow a proper connection from your iPhone to your Mac to run the app locally. **Make sure that the developer mode is enabled on your iPhone device, and iOS is updated properly.**
### Installing Requirements
For all the projects to be compatible, make sure your Python version is between 3.8 and 3.12 when creating your virtual environment. Make sure that the virtual environment is activated before installing the requirements.
```console
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Activating the Server
Make sure the virtual environment is activated before activating the Flask server.
```console
source venv/bin/activate
cd app
python connect.py
```
### Running the App
Connect your iPhone to your Mac computer or laptop with a charger. Make sure that Developer Mode is turned on in Settings > Privacy. Once your phone is configured for usage, run the front end code on the XCode IDE. **Make sure that the server has been activated.** Configure the URL on XCode to match the IP address of your computer.
```
// this line should be directly underneath the uploadImage function on Xcode/Swift
guard let url = URL(string: "http://{your computer's IP address}:5050/upload") else { return }
```
RecycLENS should now be ready to run!
