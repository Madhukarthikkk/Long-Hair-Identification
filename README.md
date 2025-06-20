Long Hair-Based Gender Prediction with Age Filtering  
This project uses machine learning to predict hair length and estimate age from facial images. Based on age and hair length, it predicts gender using custom logic:  

People aged 20 to 30: Gender is predicted based on hair length, with long hair indicating female and short hair indicating male.  

People outside the age range of 20 to 30: Gender is predicted using a separate machine learning model.  

Dataset  
The project uses the (https://drive.google.com/drive/folders/1M_LPg52KuGahlnsa0kmeFUQpBGJA00ll) dataset, which contains over 36000 celebrity images. For training and testing: 

We selected a subset of images and their attribute labels.  
During training, 2,000 test images are automatically saved to the test_images/ folder for GUI predictions.  

Models Used  
Model Type    Purpose                           Algorithm  
CNN Model    Hair Length Classification      TensorFlow Keras Sequential  
CNN Model    Age Estimation                  TensorFlow Keras Sequential  

Features :
Classifies hair as long or short.  
Predicts age from the face image.  
Determines gender based on age and hair logic.  
The GUI built with Tkinter automatically predicts a random image from the test set.  

Project Structure  
LongHairGenderDetection/  
│  
├── model_training/  
│   ├── hair_length_classifier_model.h5  
│   ├── age_model.h5  
│   ├── hair_length_training.ipynb  
│   └── age_model_training.ipynb  
│  
├── test_images/  
│   └── [2,000 random test images created during training]  
│  
├── gui/  
│   ├── hair_length_gui.py  
│   └── hair_length_age_gui.py  
│  
├── requirements.txt  
└── README.md  

How to Run  
Clone the repository and install the required libraries:  

pip install -r requirements.txt  
Download the dataset from Google Drive.  

Place images in:  
data/img_align_celeba/  

Train the models using the Jupyter notebooks:  
Run model_training/hair_length_training.ipynb  
Run model_training/age_model_training.ipynb  

Run the GUI apps:  

For hair length only:  
python gui/hair_length_gui.py  

For age and hair logic:  
python gui/hair_length_age_gui.py  
