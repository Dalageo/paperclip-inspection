<div align="center">
  <img src="https://github.com/user-attachments/assets/03bac365-e1fe-421f-86de-3855b2d03bbc" width="650" />
</div>

<div align="center">
  <a href="https://www.python.org/downloads/release/python-3127/" target="_blank">
  <img src="https://img.shields.io/badge/Python-3.12.7-blue.svg" alt="Python 3.12.7"></a>
  <a href="https://pytorch.org/get-started/locally/" target="_blank">
    <img src="https://img.shields.io/badge/PyTorch-2.5.1-orange.svg" alt="PyTorch 2.5.1"></a>
  <a href="https://developer.nvidia.com/cuda-12-4-0-download-archive" target="_blank">
  <img src="https://img.shields.io/badge/CUDA-12.4-brightgreen.svg" alt="CUDA 12.4"></a>
<a href="https://developer.nvidia.com/cudnn" target="_blank">
  <img src="https://img.shields.io/badge/cuDNN-9.1.0-brightgreen.svg" alt="cuDNN 9.1.0"></a>
  <a href="https://github.com/Dalageo/PaperClipInspection/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-AGPL%20v3-800080" alt="License: AGPLv3"></a>
  <img src="https://img.shields.io/github/stars/Dalageo/PaperClipInspection?style=social" alt="GitHub stars">
</div>

# Analyzing Paper Clips Using Deep Learning and Computer Vision Techniques📎

In this project, a combination of object detection and computer vision methods was utilized to inspect paper clips for defects. For normal clips, additional steps were taken to extract physical characteristics, such as size categorization and angle estimation. Moreover, To establish a reference for accurately converting pixel measurements to actual centimeters, an [ArUco marker](https://docs.opencv.org/3.4/d5/dae/tutorial_aruco_detection.html) with ID 0 and a size of 50x50 mm was used.

For object detection and classification part, a pretrained [YOLOv11-OBB](https://github.com/ultralytics/ultralytics) (Oriented Bounding Box) model provided by [Ultralytics](https://docs.ultralytics.com/) was fine-tuned. Since the clips appeared in various orientations, the OBB model was chosen over a standard YOLOv11 to handle rotated bounding boxes and accurately detect clips, even when rotated. Annotations were prepared using [CVAT](https://www.cvat.ai/), which supports rotated annotations, while to further enhance the model’s performance, data augmentation techniques like rotation, scaling, flipping, and brightness adjustments were applied.

The pipeline for this project begins by detecting the Aruco marker to establish the scale for real-world measurements. The YOLOv11-OBB model then identifies and classifies the objects in the frame. If a paper clip is classified as normal, additional computer vision techniques are used to determine its dimensions, categorize it into one of three predefined size categories, and calculate its angle relative to the frame. All of these operations are performed in real-time, with results displayed on the video feed. 

This project demonstrates how computer vision and deep learning techniques can be integrated to create a robust automated inspection system. With modifications and of course new data collection, this system can also be adapted to inspect different objects, providing a practical solution for quality control in various manufacturing environments where consistency and accuracy are essential.

# Dataset Description 

The [PaperClip](https://www.kaggle.com/datasets/dalageo/paperclip?select=test) dataset for this project was captured using a [Razer Kiyo X Web Camera Full HD](https://www.razer.com/eu-en/streaming-cameras/razer-kiyo-x?srsltid=AfmBOoqMjKnf8ZXzLBUvIMu-k8BlWMwC23C2Wi3-WU_g-Jw_I4G7_2zL) and includes 80 high-resolution images (1920x1080). These images showcase standard paper clips in a variety of configurations, capturing both normal and defective conditions across three different sizes: 24x7 mm, 32x9 mm, and 44x11 mm. Below is an image that illustrates the range of sizes along with the ArUco marker used in the project:

<div align="center">
  <img src="https://github.com/user-attachments/assets/e1d406c5-84ab-487b-b467-318b38a07cd0" alt="Paper Clips" width="750">
</div>

To enhance dataset variability and provide the potential for creating a more robust model, both types of paper clips (normal and defected) were captured in various positions and orientations. Additionally, the defective clips included various defects, such as bending, twisting, or asymmetrical warping. Overall, the dataset consists of 80 images featuring a total of 247 paper clips—127 normal and 120 defective. As a result, each image may contain multiple clips rather than just one. To achieve balanced training, the dataset is divided into two parts: the training set includes 68 images depicting 110 normal clips and 107 defective ones, while the validation set contains 12 images with 17 normal and 17 defective clips, making up approximately 15% of the total dataset. The table below summarizes the distribution of normal and defective clips across the training and validation sets, along with the total number of images in each set.

<div align="center">
  <table>
    <tr>
      <th></th>
      <th>Normal Clips</th>
      <th>Defective Clips</th>
      <th>Total Images</th>
    </tr>
    <tr>
      <td><b>Training Set</b></td>
      <td>110</td>
      <td>107</td>
      <td>68</td>
    </tr>
    <tr>
      <td><b>Validation Set</b></td>
      <td>17</td>
      <td>17</td>
      <td>12</td>
    </tr>
  </table>
</div>

In addition to the training and validation sets, a 23-second test video (approximately 690 frames) in .mp4 format, captured at the same resolution, is provided for users who wish to test their model, since obtaining these specific types and sizes of paper clips may be challenging. The video features various paper clips, both normal and defective, with some defects not included in the training or validation datasets. This makes it ideal for evaluating the model’s capabilities and provides a realistic test of its performance in real-world scenarios.

## Setup Instructions

### <img src="https://github.com/user-attachments/assets/8d36d1a5-e9b1-40d1-97c9-3d4ca49e9c95" alt="Local PC" width="18" height = "16" /> **Local Environment Setup**

1. **Clone the repository**:
   ```sh
   git clone https://github.com/Dalageo/paperclip-inspection.git

2. **Navigate to the cloned directory**:
   ```sh
   cd PaperClipInspection
  
3. **Open the `Analyzing Paper Clips Using Deep Learning and Computer Vision Techniques.ipynb` using your preferred Jupyter-compatible environment (e.g., [Jupyter Notebook](https://jupyter.org/), [VS Code](https://code.visualstudio.com/), or [PyCharm](https://www.jetbrains.com/pycharm/))**
   
4. **Update the `best_yolo` variable to point to the location of the `PaperClipInspection-YOLOv11-OBB.pt` model on your local environment.**
   
5. **Run the cells sequentially to reproduce the results in real-time.**

*You can reproduce the results of this project in real-time using your own camera **only if you have the exact same types and sizes of paper clips used in this project**. If you don’t have these specific paper clips, you can test the model using the provided .mp4 video in the `test` folder. Alternatively, you can collect a new dataset, retrain the model, and modify the code accordingly. To run YOLOv11 on the GPU, you will need to activate GPU support based on your operating system and install the required dependencies. You can follow this [guide](https://pytorch.org/get-started/locally/) provided by [PyTorch](https://pytorch.org/) for detailed instructions.*

## Acknowledgments

Special thanks to [Ultralytics](https://github.com/ultralytics/ultralytics) for providing the pretrained YOLOv11 model for educational purposes, as well as to the [CVAT](https://github.com/cvat-ai/cvat) community for their user-friendly and free annotation software. Both were essential to the development of this project.

<div align="center">
  <br>
  <a href="https://www.ultralytics.com/">
    <img src="https://github.com/user-attachments/assets/872182a7-96db-46c2-ad4d-a2ef108e2e51" alt="Ultralytics" width="200"/></a>
  <a href="https://www.cvat.ai/">
    <img src="https://github.com/user-attachments/assets/9a447784-e8eb-4e01-a3c5-c081db4d0693" alt="CVAT" width="150"/></a>
</div> 

## License
The provided fine-tuned model, along with the dataset, notebook, and accompanying documentation, are licensed under the [AGPL-3.0 license](https://www.gnu.org/licenses/agpl-3.0.en.html). This license was chosen to promote open collaboration, ensure transparency, and allow others to freely use, modify, and contribute to the work, while maintaining consistency, as the provided pretrained YOLOv11 model is also licensed under AGPL-3.0. For more information on the licensing and access to YOLOv11 models, please visit the [YOLOv11 repository](https://github.com/ultralytics/ultralytics).

Any modifications or improvements must also be shared under the same license, with appropriate acknowledgment.

<div align="center">
  <br>
  <a href="https://www.gnu.org/licenses/agpl-3.0.en.html">
    <img src="https://github.com/user-attachments/assets/f3c6face-aa86-45da-8d20-d8ae25e49e28" alt="AGPLv3-Logo" width="200">
  </a>
</div>




