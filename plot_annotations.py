import os
import json
import cv2
import glob


split = 'train'

inp_vid_folder = 'roadpp/videos/'
out_vid_folder = 'roadpp/vis_vids'

def set_video(inp, video_name):
    print(inp)
    cap = cv2.VideoCapture(inp)
    ret, frame = cap.read()
    frheight, frwidth, ch = frame.shape
    print(frheight,'.......',frwidth)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    video_width = frwidth
    video_height = frheight
    size = (video_width, video_height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, video, fps,no_frames


for j_vid in glob.glob('combined_jsons/'+split+'/*.json'):
    print(j_vid)
    with open(j_vid, 'r') as f:
        j_vid_json = json.load(f)
    j_vid_name = os.path.basename(j_vid)[:-5]
    
    inp_vid_file,out_vid_file,fps,no_frames = set_video(inp_vid_folder+'/'+split+'/'+j_vid_name,out_vid_folder+'/'+split+'/'+j_vid_name)
    inp_vid_file.set(2,0)
    for i in range(1,no_frames+1):        
        ret, frame = inp_vid_file.read()
        if str(i) in j_vid_json['frames']:
            frame_annos = j_vid_json['frames'][str(i)]

            
            for anno in frame_annos:
                labs = anno['tags']
                bbox = [int(anno['box']['x1']),int(anno['box']['y1']),int(anno['box']['x2']),int(anno['box']['y2'])]
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2],bbox[3]), (0, 255, 0), 2)

                for count,lab in enumerate(labs):
                    cv2.putText(frame, lab, (bbox[0], bbox[1]+count*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
                
        # cv2.imwrite('out.png',frame)
        out_vid_file.write(frame)


    out_vid_file.release()









