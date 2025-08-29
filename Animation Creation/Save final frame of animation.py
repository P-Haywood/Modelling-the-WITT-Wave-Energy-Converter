# save final frame
import matplotlib.pyplot as plt
import imageio
video_path = "02 Modelling/03 Summer Delivery/Animations/Single WITT/Optimised_moored_0.3Hz_200s.mp4"
video = imageio.get_reader(video_path, 'ffmpeg')
last_frame = None
for frame in video:
    last_frame = frame  # keeps updating until the last frame
imageio.imwrite("02 Modelling/03 Summer Delivery/Animations/Single WITT/Optimised_moored_final_frame_0.3Hz.png", last_frame)  # save as PNG
plt.imshow(last_frame)
plt.axis('off')
plt.show()