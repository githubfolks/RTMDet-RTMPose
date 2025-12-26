# Running on Google Colab

This project is ready to run on Google Colab's free GPU instances.

## Steps

1.  **Compress the Project:**
    Zip the entire `RTMDet-RTMPose` folder.
    *(Make sure to include `custom_dataset` inside it).*

2.  **Upload to Drive:**
    Upload the folder to your **Google Drive** root directory.
    Expected Path: `My Drive/RTMDet-RTMPose`

3.  **Open the Notebook:**
    - Go to [Google Colab](https://colab.research.google.com/).
    - Click `File` -> `Upload notebook`.
    - Upload the `Run_on_Colab.ipynb` file from this folder.

4.  **Configure Runtime:**
    - In Colab, go to `Runtime` -> `Change runtime type`.
    - Set Hardware accelerator to **T4 GPU** (or better).

5.  **Run:**
    - Execute the cells in order.
    - Check the output logs for training progress.
    - Weights will be saved to your Google Drive automatically inside `RTMDet-RTMPose/train/weights/`.
