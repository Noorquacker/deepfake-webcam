import ffmpeg, time
import numpy as np

# The following imports were from upstream
import os
import yaml
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp


# do your AI initialization here
source_image = imageio.imread('REPO/ningen.jpg')
cpu = False
relative = True
adapt_movement_scale = True
source_image = resize(source_image, (256, 256))[..., :3]
# load checkpoints but hardcoded lmao
def load_checkpoints():
	with open('config/vox-adv-256.yaml') as f:
		config = yaml.load(f)
	generator = OcclusionAwareGenerator(**config['model_params']['generator_params'], **config['model_params']['common_params'])
	if not cpu:
		generator.cuda()
	kp_detector = KPDetector(**config['model_params']['kp_detector_params'], **config['model_params']['common_params'])
	if not cpu:
		kp_detector.cuda()
	if cpu:
		checkpoint = torch.load('REPO/vox-adv-cpk.pth.tar', map_location=torch.device('cpu'))
	else:
		checkpoint = torch.load('REPO/vox-adv-cpk.pth.tar')
	generator.load_state_dict(checkpoint['generator'])
	kp_detector.load_state_dict(checkpoint['kp_detector'])
	if not cpu:
		generator = DataParallelWithCallback(generator)
		kp_detector = DataParallelWithCallback(kp_detector)
	generator.eval()
	kp_detector.eval()
	return generator, kp_detector


# NOW we can start the AI stuff
print('Loading Torch...', flush=True)
with torch.no_grad():
	source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
	if not cpu:
		source = source.cuda()
	else:
		torch.set_num_threads(12)
	print('Loading checkpoints...', flush=True)
	generator, kp_detector = load_checkpoints()
	kp_source = kp_detector(source)
	kp_driving_initial = False
	print('Opening webcams...', flush=True)
	inProc = ffmpeg.input('/dev/video0', loglevel='fatal', r=10, crop='480:480:80:0').vflip().hflip().output('pipe:', format='rawvideo', pix_fmt='rgb24', s='256x256').run_async(pipe_stdout=True)
	outProc = ffmpeg.input('pipe:', loglevel='fatal', format='rawvideo', pix_fmt='rgb24', s='256x256').output('/dev/video3', format='v4l2', s='640x480').overwrite_output().run_async(pipe_stdin=True)

	# Testing only
	bypass = False
	#bypass = ffmpeg.input('pipe:', loglevel='fatal', format='rawvideo', pix_fmt='rgb24', s='256x256').output('/dev/video4', format='v4l2', s='640x480').overwrite_output().run_async(pipe_stdin=True)

	prev_time = time.time()
	print('Warming up...', flush=True)

	# My camera takes a whopping 2 seconds to warm up are you kidding me
	inProc.stdout.read(256 * 256 * 3)
	os.system('v4l2-ctl -c exposure_auto=1')
	os.system('v4l2-ctl -c exposure_absolute=3000')
	while time.time() - prev_time < 0:
		in_bytes = inProc.stdout.read(256 * 256 * 3)
	while True:
		in_bytes = inProc.stdout.read(256 * 256 * 3)
		if bypass:
			bypass.stdin.write(in_bytes)
		if not in_bytes:
			continue
		inFrame = np.frombuffer(in_bytes, np.uint8).reshape([1, 3, 256, 256], order='F').swapaxes(2, 3) / float(255)
		driving_frame = torch.tensor(inFrame, dtype=torch.float32)
		if not kp_driving_initial:
			kp_driving_initial = kp_detector(driving_frame)
		if not cpu:
			driving_frame = driving_frame.cuda()
		kp_driving = kp_detector(driving_frame)

		# Oh boy, here we go
		kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving, kp_driving_initial=kp_driving_initial, use_relative_movement=relative, use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
		out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
		outFrame = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0].copy(order='C')
		# Write the predicted garbage to the virtual webcam
		outProc.stdin.write(img_as_ubyte(outFrame))

		# Frametime info
		print(f'Frametime: {int((time.time() - prev_time) * 1000)}ms')
		prev_time = time.time()

outProc.stdin.close()
