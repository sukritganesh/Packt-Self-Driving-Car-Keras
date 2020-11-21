###### ONE TIME ONLY ######
conda create --name sg_tf python=3.7 numpy matplotlib
conda install -c anaconda flask
conda install -c conda-forge python-socketio
conda install -c conda-forge eventlet
conda install -c anaconda pillow
conda install -c conda-forge opencv

###### BEFORE USE ######
conda activate sg_tf
cd Documents/LearningProgramming/Python/PacktSelfDriving_1/self-driving-car/BehaviouralCloning
python drive.py

<!-- conda install -c conda-forge tensorflow
conda install -c conda-forge keras -->
