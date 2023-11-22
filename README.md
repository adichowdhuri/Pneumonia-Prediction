<!DOCTYPE html>
<html>
<head>
</head>
<body>

<h1>Pneumonia Prediction Using Chest X-Rays</h1>

<h1>Introduction</h1>
<p>This repository is dedicated to a machine learning project that aims to predict pneumonia from chest X-ray images using Convolutional Neural Networks (CNN).</p>

<h1>Project Overview</h1>
<ul>
    <li><strong>Objective</strong>: To develop a reliable model for pneumonia detection from chest X-ray images.</li>
    <li><strong>Methodology</strong>: Utilizing CNN for efficient and accurate image classification.</li>
    <li><strong>Dataset</strong>: A comprehensive collection of chest X-ray images.</li>
</ul>

<h1>Repository Structure</h1>
<pre><code>
Pneumonia-Prediction/
│
├── Models/                  # Contains different CNN models used
│   ├── InceptionV3model.ipynb
│   ├── ResNet-18model.ipynb
│   └── VGG16model.ipynb
│
├── chest_xray/              # Dataset directory with X-ray images
│   ├── test/
│   ├── train/
│   └── val/
│
├── training_logs/           # Logs from training sessions
│
├── .devcontainer/           # Configuration for development containers
│
├── app.py                   # Main application script for model deployment
│
├── requirements.txt         # List of dependencies
│
└── README.md                # Project documentation
</code></pre>

<h1>Tech Stack</h1>
<p>The Pneumonia Prediction project utilizes a range of technologies and libraries for machine learning, data processing, and application development. Below is a list of the key components of our tech stack:</p>

<ul>
    <li><img src="https://icons.iconarchive.com/icons/papirus-team/papirus-apps/256/python-icon.png" alt="Python Icon" style="width:20px;height:20px; vertical-align:middle;"/> <strong>Python:</strong> The primary programming language used for developing the machine learning model and processing data.</li>
    <li><img src="https://static-00.iconduck.com/assets.00/tensorflow-icon-955x1024-hd4xzbqj.png" alt="TensorFlow Icon" style="width:20px;height:20px; vertical-align:middle;"/> <strong>TensorFlow:</strong> An open-source machine learning library used for building and training the neural network models.</li>
    <li><img src="https://numpy.org/images/logo.svg" alt="NumPy Icon" style="width:20px;height:20px; vertical-align:middle;"/> <strong>NumPy:</strong> A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.</li>
    <li><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Matplotlib_icon.svg/2048px-Matplotlib_icon.svg.png" style="width:20px;height:20px; vertical-align:middle;"/> <strong>Matplotlib:</strong> A plotting library for the Python programming language and its numerical mathematics extension NumPy.</li>
    <li><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/883px-Jupyter_logo.svg.png" style="width:20px;height:20px; vertical-align:middle;"/> <strong>Jupyter Notebook:</strong> An open-source web application for creating and sharing documents with live code, equations, and visualizations.</li>
</ul>



<h1>Getting Started</h1>

<h2>Prerequisites</h2>
<ul>
    <li>Python 3.x</li>
    <li>pip package manager</li>
    <li>ipykernel</li>
</ul>

<h2>Installation</h2>
<h3>Cloning the Repository</h3>
<ol>
    <li><strong>Open a Terminal or Command Prompt</strong>: 
        <ul>
            <li>On Windows, search for <code>Command Prompt</code> in the Start menu.</li>
            <li>On macOS or Linux, open the <code>Terminal</code>.</li>
        </ul>
    </li>
    <li><strong>Navigate to Your Desired Directory</strong>: 
        <ul>
            <li>Use <code>cd</code> to change directories, e.g., <code>cd Desktop</code>.</li>
        </ul>
    </li>
    <li><strong>Clone the Repository</strong>:
        <ul>
            <li>Run the following command:
                <pre><code>git clone https://github.com/adichowdhuri/Pneumonia-Prediction.git</code></pre>
            </li>
            <li>This will create a <code>Pneumonia-Prediction</code> folder in your current directory and download the repository contents into it.</li>
        </ul>
    </li>
</ol>

<h3>Installing Dependencies</h3>
<ol>
    <li><strong>Navigate to the Repository Directory</strong>:
        <ul>
            <li>Change to the cloned repository's directory:
                <pre><code>cd Pneumonia-Prediction</code></pre>
            </li>
        </ul>
    </li>
    <li><strong>Ensure Python is Installed</strong>:
        <ul>
            <li>Run <code>python --version</code> or <code>python3 --version</code>. This should display the Python version.</li>
            <li>If Python is not installed, download it from <a href="https://www.python.org/downloads/">python.org</a>.</li>
        </ul>
    </li>
    <li><strong>Install Pip (if not installed)</strong>:
        <ul>
            <li>Pip is the Python package manager. Follow the installation instructions on the <a href="https://pip.pypa.io/en/stable/installation/">pip documentation page</a>.</li>
        </ul>
    </li>
    <li><strong>Install Project Dependencies</strong>:
        <ul>
            <li>Make sure you are in the directory containing <code>requirements.txt</code>.</li>
            <li>Run:
                <pre><code>pip install -r requirements.txt</code></pre>
            </li>
            <li>This command installs all the Python libraries listed in <code>requirements.txt</code>.</li>
        </ul>
    </li>
</ol>

<h2>Enabling GPU Acceleration</h2>
<p>To utilize GPU acceleration, follow these steps:</p>

<h3>Install CUDA Toolkit</h3>
<ul>
    <li>Download and install the appropriate version of <a href="https://developer.nvidia.com/cuda-downloads">CUDA Toolkit</a>.</li>
</ul>

<h3>Install cuDNN</h3>
<ul>
    <li>Download <a href="https://developer.nvidia.com/cudnn">cuDNN</a> corresponding to your CUDA Toolkit version.</li>
    <li>Follow the installation guide provided by NVIDIA.</li>
</ul>

<p>You can use Anaconda to download both the <code>cudatoolkit</code> and <code>cudnn</code> using</p>
<pre><code>conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0</code></pre>

<h3>Install TensorFlow</h3>
<ul>
    <li>For TensorFlow versions below 2.11, compatible with certain CUDA and cuDNN versions:
        <pre><code>pip install tensorflow==2.10.0</code></pre>
    </li>
</ul>

<h1>Loading the Trained Model</h1>
<p>To use the trained pneumonia prediction model in your own projects, follow these steps:</p>

<ol>
    <li><strong>Download the Trained Model</strong>:
        <p>First, download the trained model file. This file should be available in the repository or via a provided link.</p>
    </li>
    <li><strong>Import Necessary Libraries</strong>:
        <p>Ensure you have the necessary libraries installed, such as TensorFlow. You can install them using pip:</p>
        <pre><code>pip install tensorflow</code></pre>
    </li>
    <li><strong>Load the Model</strong>:
        <p>Use TensorFlow to load the model. Here's an example of how to do it:</p>
        <pre><code>import tensorflow as tf
model = tf.keras.models.load_model('path/to/model.h5')</code></pre>
        <p>Replace 'path/to/model.h5' with the actual path to the downloaded model file.</p>
    </li>
    <li><strong>Use the Model for Prediction</strong>:
        <p>Once the model is loaded, you can use it to make predictions on new chest X-ray images:</p>
        <pre><code>prediction = model.predict(image_data)</code></pre>
        <p>Here, 'image_data' should be the image you want to analyze, processed as required by the model.</p>
    </li>
</ol>


<h1>Models Used</h1>
<h2>InceptionV3</h2>
<p>A deep CNN model known for its efficiency in image classification, particularly useful for complex image recognition tasks.</p>
<img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs13634-021-00740-8/MediaObjects/13634_2021_740_Fig1_HTML.png" alt="InceptionV3 Model" >

<h2>ResNet-18</h2>
<p>Part of the ResNet family, this model uses residual learning to facilitate training deeper networks.</p>
<img src="https://www.frontiersin.org/files/Articles/590371/fnbot-14-590371-HTML/image_m/fnbot-14-590371-g007.jpg" alt="ResNet-18 Model">

<h2>VGG16</h2>
<p>A deep CNN known for its depth and simplicity, widely used in image recognition tasks.</p>
<img src="https://miro.medium.com/v2/resize:fit:1400/1*NNifzsJ7tD2kAfBXt3AzEg.png" alt="VGG16 Model" width="1000" height="400">

<h1>Contributing</h1>
<p>We welcome contributions! Please fork the repository, make your changes, and submit a pull request.</p>

<h1>License</h1>
<p>This project is licensed under the <a href="LICENSE">MIT License</a>.</p>

<h1>Acknowledgements</h1>
<p>Special thanks to all contributors and the open-source community for their support and resources.</p>

</body>
</html>
