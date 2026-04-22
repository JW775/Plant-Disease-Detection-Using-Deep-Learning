# Plant-Diease-Detection Using deep-learning

This is a deep learning-based project for plant leaf disease recognition, which aims to **automatically identify the health status and specific disease types of plant leaves** using convolutional neural networks (CNNs), providing an **intelligent auxiliary diagnosis tool** for agricultural production.
---
## Functional Featyres
 - **Multi-class Recognition**: Supports classification of various diseases and healthy status for multiple plants (e.g., tomato, potato, apple, etc.).
 - **Transfer Learning**: Uses the ResNet18 model pre-trained on ImageNet for transfer learning to improve training efficiency and accuracy.
 - **Web Interface**: Provides a simple Flask-based Web interface that supports image upload for real-time prediction and result visualization.
 - **Modular Design**: Separates training, prediction, and Web service code with a clear structure for easy maintenance and extension.
 - **Team Collaboration Friendly**: Separates training, prediction, and Web service code with a clear structure for easy maintenance and extension.
---
## Tech Stack
- **Program Language**: Python 3.8+
- **Deep Learning Framework**: Flask
- **Core Dependencies**: torch, torchvision, numpy, pandas, opencv-python, pillow, flask
----
## Environment Setup
- Before you begin, please ensure that Python 3.8 or a later version is installed on your computer.
1. Clone the project repository
Open the terminal and run the following command to clone the project locally
```bash
git clone <https://github.com/JW775/Plant-Disease-Detection-Using-Deep-Learning.git>
cd Plant-Disease-Detection-Using-Deep-learning
```
2. Create a virtual environment
```bash
conda create -n plant_env python = 3.8
```
Windows:
```bash
conda active plant_env
```
Mac / Linux
```bash
conda active plant_env
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. After installation, you can start the application with the following command:
```bash
python app.py
```
---
## Project Structure
```plaintext
|--- data/						# Dataset storage 
|	|--- train/					# Training set ( classified by folder )
|	|--- val/					# Validation set
|	|--- test/					# Test set
|
|--- models/					# Trained model weights ( .pth files )
|	|--- best_model.pth	        
|	
|--- static/					# Static files for Web application ( for Flask )
|	|--- images/				# Temporarily uploaded images by users
|	|--- cas/ 					# Style sheets
|
|--- templates/					# HTML web templates ( for Flask )
|	|--- index.html 			# Homepage
|	|--- result.html			# Result display page
|	
|--- src/						# Core source code directory
|	|--- __init__.py			
|	|--- dataset.py				# Data loading and preprocessing ( Dataset & Dataloader )
|	|--- model.py				# Model definition ( ResNet18, etc. )
|	|--- train.py				# Training script
|	|--- utils.py				# Utility functions (plotting, accuracy calculation, etc. )
|
|---.gitignore					# Git ignore file configuration
|--- app.py						# Flask Web application entry
|--- predict.py					# Single image prediction script ( for command line )
|--- README.md					# Project description document
|--- requiremeents.txt			# Dependencies list ( pip freeze > requirements.txt )
|--- LICENSE					# Open source license


```
---
## Team Workflow
1. After cloning the repository, navigate into the downloaded floder to proceed.
```bash
cd Plant-Disease-Detection-Using-Deep-Learning
```
2. Create and switch to your branch
- Never modify code directly on the main branch!
```bash
git checkout -b -feature/data-augmentation-xiaoming (eg.)
```
3. Start work and Save changes
```bash
# check what changes
git tatus

# Add all modified files to the staging area
git add 

# Commit your changes (clearly describe what you have done)
git commit -m <What you changes>
```
4. Push to remote
```bash
git push origin <feature/data-auhentation-xiaoming>
```
5. Create a pull request
- You need to go to the GitHub webpage and click **"Compare & pull request"**.

---
Team Nembers
- JiaYi Wu: Documentation and Testing
- XiaoLei Luan: Web application Development (Backend Development)
- YiXuan Liu: Interface Design (Frontend Development)
- Zheng Wang: Dataset Construction (Data processing)
- Nan Xiang: Deep Learning Algorithm (Model Training)



