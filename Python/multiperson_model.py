import sys
sys.path.append('.')
import shutil
from VIBE.multi_person_tracker import MPT
from VIBE.multi_person_tracker.data import video_to_images
from VIBE.multi_person_tracker import save_model

def main():
    vf = sys.argv[1]

    image_folder = video_to_images(vf)
    #We create the model, using the method MPT from VIBE library
    mot = MPT(
        display=True,
        detector_type='yolo',  # 'maskrcnn'
        batch_size=10,
        detection_threshold=0.7,
        yolo_img_size=416,
    )
    #The output of the multiperson tracker is an object that contains each person segmented, over the frames of the video.
    result = mot(image_folder, output_file='sample.mp4')
    for detection in result.nd_arr:
        save_model(detection, output_folder, output_file=result.index)
    #for each person detected, we save the model that can be later used with mesh extractor, for obtaining a model of the segmented person.
    
    shutil.rmtree(image_folder)
    #we delete the tmp files

if __name__ == '__main__':
    main()
