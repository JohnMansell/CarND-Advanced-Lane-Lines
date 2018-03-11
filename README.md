# Read Me
# Advanced Lane Finding
### John Mansell

This is my project submission for the Advanced Lane Finding Project from Udacity's Self Driving Car Nano Degree

# Udacity Reviewer
> Project Submission - 3/12/2018  
> Pipeline = [my_video_gen.py](my_video_gen.py) and [HelperFunctions.py](helper_functions.py)  
> Writeup = [WriteUp.ibynb](WriteUp.ipynb)  
> Video = [output_videos/output_project_video.mp4](output_videos/output_project_video.mp4)  

### Files
[**cam_cal.py**](cam_cal.py) -- Callibrates the camera using the findChessBoardCorners technique from Open CV
[**helper_functions.py**](helper_functions.py) -- Functions used throughout the project:
> * find_chessboard_corners()
> * color_threshold()
> * ABS_sobel()
> * mag_threshold()
> * dir_threshold()
> * window_mask()
> * show()
> * get_transform()
> * warp_image()
> * create_binary()
> * process_image()
> * straight_line_template
> * pattern_fit()

[**image_gen.py**](image_gen.py) -- original pipeline for processing images  
[**my_video_gen.py**](my_video_gen.py) -- pipeline for processing input videos  
[**ReadMe.md**](ReadMe.md) -- for GitHub -- description of files in the project  
[**Scrath_Pipeline.ipynb**](scratch_pipeline.ipynb) -- jupyter notebook which I used for testing out different parts of the pipeline.  
[**tracker.py**](tracker.py) -- tracker class for finding lane lines  
[**WriteUp.ibynb**](WriteUp.ipynb) -- Write up report for Udacity project submission  

## Folders
> * **.ipynb_checkpoints** -- saved checkpoints for the jupyter notebooks
> * **camera_cal** -- Images used to calibrate the camera
> * **input_videos** -- Input Videos for the project
> * **output_images** -- Images at different stages of the pipeline, mostly used in the writeup notebook
> * **output_videos** -- Final output videos for the project:
>   * output_project_video
>   * output_challenge_video
>   * output_harder_challenge_video
> * **post_calibration** -- Chessboard images post camera calibration
> * **test_images** -- Example pictures of the road used as a starting point for building the pipeline
