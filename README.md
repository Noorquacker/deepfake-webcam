# Real-Time Deepfake Webcam

Based on the [First Order Motion Model for Image Animation](https://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation), this code is forked from their [wonderful code repository](https://github.com/AliaksandrSiarohin/first-order-model).

## Prerequisites

In order to run in real-time, you _need_ a GTX 1060 or above in order to go above 20 FPS and look somewhat decent. This repo was tested on a GTX 1060 6gb and it worked at 20-25 FPS.

You also _must_ be running on Linux if you want to use a live webcam. We do not have a cross-platform way of having virtual webcams, but [you can help](https://github.com/Noorquacker/deepfake-webcam/pulls).

### Installation

1. As root (which is not advised, but do what you want), do what you do on every other Python repo:  
	```bash
	sudo pip3 install -r requirements.txt
	```
2. Make a new folder called `REPO` and put the [vox model checkpoints](https://drive.google.com/open?id=1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH) in there (download `vox-cpk.pth.tar` and `vox-adv-cpk.pth.tar` and move them in there)
3. Install and load the [v4l2loopback](https://github.com/umlaeute/v4l2loopback#DISTRIBUTIONS) kernel module for your distro
4. Create a fake webcam device. Instructions can be found in v4l2loopback's documentation
5. Edit `webcam.py` to point to the proper input and output webcams
6. Run `webcam.py` as root, unless you have permissions set up nicely
7. Show up to Zoom meetings
