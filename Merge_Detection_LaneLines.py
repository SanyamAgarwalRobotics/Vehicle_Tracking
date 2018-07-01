import VehicleDetectionTracking
import Advance_LaneLines
from moviepy.editor import VideoFileClip

def mergeProjectPipelines(img):
    Detection_img = VehicleDetectionTracking.processVDT_image(img)
    return Advance_LaneLines.process_image(Detection_img)

######################################################################################################################
#  Following section creates the video
######################################################################################################################
def createOutVideo():
    white_output = 'project_result_combined.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(mergeProjectPipelines)
    white_clip.write_videofile(white_output, audio=False)
    return white_clip

# Reset the data structures used in Advanced lane detection and Vehicle tracking to start with fresh data
#advancedLaneDetect_Pipeline.lastFrameData.resetLaneData()
#VehicleDetectionAndTracking.heatmaps.clear()
# Create Video
white_clip = createOutVideo()