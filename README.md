# ROAD-NEEYAM-RAKSHAK
Many families lose someone very close to some road accidents which are sometimes caused
              by DROWSINESS (due to tiredness ), NOT wearing SEATBELTS  and Sometimes
              due to some kids(MINOR) or people who are NOT AUTHORISED to
              drive a car (i.e do NOT have a DRIVING LICENCE) . In order to maintain the drivers
              safety , and to ensure that proper Road laws are followed , ROAD-नियम RAKSHAK (सारथी) has been developed that can check drowsiness and authenticity of Driver in REAL-TIME that will alert
              them when they are yawning or feeling sleepy , it will also detect if the driver has worn a seatbelt or not . Any violation of road safety rules like drowsy (yawn + sleep) driving and No seatbelt will result into penalty because when you break the rule your picture will be clicked as a proof . At the same time you should be a registered Driver because , only REGISTERED Drivers are recognized and if you are UNREGISTERED then your picture will be clicked and saved in the folder 

 these are the basic 4 features of my app 

 1) detects sleep by calculating EAR 
    if (Eyes aspect ratio <0.3)
      SLEEP DETECTED // EYES ARE CLOSED 
 2) detects yawn by calculating MAR 
    if mouth aspect ratio>14
        YAWN DETECTED

 3) detects seat belt (detect edges and if a group of lines are at angles between 0.7 to 2 then seat belt detected)
 4) detects authentication of driver 
     if the driver is unregistered then face is unrecognized and hence the driver is not having driving licence that is why it is not in face_dataset 

     whenever any rule is breaken , the pic are clicked and saved in folders like DATASET that records sleep,yawn and seatbelt detection
     and UNREGISTERED DRIVERS stores photo of unregistered drivers .

     DATASET
     in the dataset folder you can see the ss clicked while i broke some rules like yawn and sleep & seatbelt detection photo also present

     Similarly in UNREGISTERED DRIVERS FOLDER 
     you can see the pic of unrecognized driver 
     

INSTALLATION GUIDE

1) DOWNLOAD ZIP FOLDER FROM GITHUB 
2) EXTRACT THE FOLDER FROM ZIPPED FOLDER
3) OPEN THE ZIPPED FOLDER IN PYCHARM 
4) VIRTUAL ENV PYTHON 3.9 , PYTHON 3.9 SHOULD BE USED AS PYTHON INTERPRETER
5) OPEN THE REQUIREMENTS.TXT FILE AND SELECT ALL PACKAGES AND INSTALL THEM BY RIGH CLICKING>INSTALL ALL
6) NO ALL PACKAGES ARE INSTALLED , YOU ARE GOOD TO GO !
7) RIGHT CLICK ON APP1.PY AND RUN THE PROJECT .
8) THE PROJECT STARTS TO RUN ON LOCAL HOST

HOPE THE PROJECTS WORKS FINE!:)

