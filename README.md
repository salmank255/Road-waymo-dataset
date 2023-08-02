# Download and pre-process ROAD++ dataset

Here we provide the **download** and **pre-processing instructions** for the ROAD++ dataset that is released through ROAD++ challenge: [The Second Workshop & Challenge on Event Detection for Situation Awareness in Autonomous Driving](https://sites.google.com/view/road-plus-plus/home) and uses [3D-RetinaNet](https://github.com/salmank255/ROAD_plus_plus_Baseline) code as a **baseline**, which also contains the evaluation code. 

The structure of our ROAD++ is exactly similar to our [ROAD dataset](https://github.com/gurkirt/road-dataset).

## Main Features

- Action annotations for human as well as other road agents, e.g. Turning-right, Moving-away etc. 
- Agent type labels, e.g. Pedestrian, Car, Cyclist, Large-Vehicle, Emergency-Vehicle etc.
- Semantic location labels of the location of agent, e.g. in vehicle lane, in right pavement etc.
- **198K** frames from **1000** videos annotated, each video is 20 seconds long on an average.
- track/tube id annotated for every bounding box on every frame for every agent in the scene.
- **54K** tubes/tracks of individual agents.
- **3.9M** bounding  box-level agent  labels.
- **4.3M** and **4.2M** bounding box-level action and location labels.

## Attribution
ROAD++ dataset was made using the [Waymo Open Dataset](https://waymo.com/open/), provided by Waymo LLC under license terms available at [waymo.com/open](https://waymo.com/open/).

By downloading or using the ROAD++ dataset and/or the Waymo Open Dataset, you are agreeing to the terms of the [Waymo Open Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms)—which includes a requirement that you only use the Waymo Open Dataset (or datasets built from it, such as the ROAD++ dataset) for noncommercial purposes. “Non-commercial Purposes" means research, teaching, scientific publication and personal experimentation. Non-commercial Purposes include use of the dataset to perform benchmarking for purposes of academic or applied research publication. Non-commercial Purposes does not include purposes primarily intended for or directed towards commercial advantage or monetary compensation, or purposes intended for or directed towards litigation, licensing, or enforcement, even in part. 


## Download

BY DOWNLOADING THE DATASET VIDEOS YOU ARE BOUNDED TO ADHERE TO PRIVACY GUIDELINES OF [WAYMO Open Dataset](https://waymo.com/open/). PLEASE VISIT [WAYMO](https://waymo.com/open/terms/) TERMS & PRIVACY POLICY FOR MORE DETAILS. THE VIDEOS FROM [WAYMO Open Dataset](https://waymo.com/open/terms/) AND PROVIDED ANNOTATIONS ARE ONLY FOR ACADEMIC PURPOSE. 

We release the annotations created by [Visual Artificial Intelligence Laboratory](https://cms.brookes.ac.uk/staff/FabioCuzzolin/), and the sub-set video (Frontal only) from [WAYMO](https://waymo.com/open/terms/). 

The videos and annotations are available in the Community Contributions section of [Waymo Open Dataset](https://waymo.com/open/) and can be downloaded from the Community Contributions section of [WAYMO Download](https://waymo.com/open/download/).


## Frame-extraction

The **baseline code** for [ROAD++](https://github.com/salmank255/ROAD_plus_plus_Baseline) uses sequences of frames as input. Once you have downloaded the videos from Waymo open website, create a folder name `road-waymo` and put the annotations under it, then create another folder named `videos` under `road-waymo` folder, and put all the videos under the folder named `videos`. Now, your folder structure should look like this:

```
    road-waymo/
        - road_waymo_trainval_v1.0.json
        - videos/
            - Train_00000.mp4
            - Train_00001.mp4
            - Train_00002.mp4
            - ........

```

Before extracting the frames, you will need to make sure that you have `ffmpeg` installed on your machine or your python should include its binaries. If you are using Ubuntu, the following command should be sufficient: `sudo apt install ffmpeg`.

You can now use `extract_videos2jpgs.py` to extract the frames. You will need to provide the path to the `road-waymo` folder as an argument:
```
python extract_videos2jpgs.py <path-to-road-folder>/road_waymo/
```

Now, the `road-waymo` directory should look like this:

```
road-waymo
├── road_waymo_trainval_v1.0.json
├── videos
    ├── Train_00000.mp4
    ├── Train_00001.mp4
    ├── ...
├── rgb-images
    ├── Train_00000
        ├── 00001.jpg
        ├── 00002.jpg
        ├── ...
    ├── Train_00001
        ├── 00001.jpg
        ├── 00002.jpg
        ├── ...
    ├── ...
        
```
## Annotation Structure

The annotations for the train and validation split are saved in single `json` file named `road_plus_plus_trainval_v1.0.json`, which is located under root directory of the dataset as it can be seen above.

The first level of `road_plus_plus_trainval_v1.0.json` contains dataset level information like classes of each label type:

- Here are all the fields: `dict_keys(['all_input_labels', 'all_av_action_labels', 'av_action_labels', 'agent_labels', 'action_labels', 'duplex_labels', 'triplet_labels', 'loc_labels', 'db', 'label_types', 'all_duplex_labels', 'all_triplet_labels', 'all_agent_labels', 'all_loc_labels', 'all_action_labels', 'duplex_childs', 'triplet_childs'])`
- `all_input_labels`: All classes used to annotate the dataset
- `label_types` :  It is list of all the label types `['agent', 'action', 'loc', 'duplex', 'triplet']`.
- `all_av_action_labels`: All classes used to annotate AV actions
- `av_action_labels`: Classes finally being used for AV actions
-  Remaining fields ending with `labels` follows the same logic and AV actions described in above line.
- `duplex_childs` and `triplet_childs` contain ids of child classes form `agent`, `action` or `location` labels to construct `duplex` or `triplet` labels, 
- `duplex` is constructed using `agent` and `action` classes.
- `event` or `triplet` is constructed  using `agent`, `action`, and `location` classes.

Finally, the `db` field contains all `frame` and `tube` level annotations for all the videos:

- To access annotation for a vides, use db['Train_00000'], where `'Train_00000'` is name of a video.
- Each video annotation comes with following fields
    - `['split_ids', 'agent_tubes', 'action_tubes', 'loc_tubes', 'duplex_tubes', 'triplet_tubes', 'av_action_tubes', 'frame_labels', 'frames', 'numf']`
    - `split_ids` contains the split id assigned this videos out of `'test', 'train', and 'val'`. 
    - `numf` is number of frames in the video.
    - `frame_labels` is AV-action class ids assigned for each frame of the videos. 
    - `frames` contains frame-level annotations
        - for the each frame of the video, e.g. for `'1'` frame, frames['1'] contains `['annotated', 'rgb_image_id', 'width', 'height', 'av_action_ids', 'annos', 'input_image_id']`
        - `annotated` is flag to indicate if frame is annotated or not.
        - `rgb_image_id` = `input_image_id` is id of physical frame extracted by ffmpeg.
        - `av_action_ids`: AV action labels
        - `annos` : contains annotations of a frame with bounding boxes with unique keys like `annos['4585'] =['b19111',]` from a frame number `'4585'`, which is unique in whole dataset.
        - `annos['b09']` has following keys `dict_keys(['box', 'agent_ids', 'loc_ids', 'action_ids', 'duplex_ids', 'triplet_ids', 'tube_uid'])`
            - `box` is normalized (0,1) bounding box coordinate with `xmin, ymin, xmax, ymax`
            - `tube_uid` id of the agent tube it belongs.
            - fields ending with `_ids` contains class ids of respective label type.
    - The fields ending `tubes` contains tube-level annotation of respective label-type. 
        - for example, `db['Train_00000']['agent_tubes']` contains tubes with fields like `['544e13cc-001-01', 'a074d1bf-001-01', 'e97b3e4c-001-01', 'edb6d66a-005-01', .........]`
        - each tube has following fields `dict_keys(['label_id', 'annos'])`
            - `label_id` is class id from respective label type.
            - `annos` is dictinary with keys made of frame_ids, e.g. `['agent_tubes']['10284a58-002-01']['annos'].keys()` >> `dict_keys(['4585', '4586', ......, '4629', '4630'])`
            - `annos['4585'] = 'b19111'` stores unique key which points to frame-level annotations in frame number '4585'.

## Evaluation

Now that you have the dataset and are familiar with its structure, you are ready to train or test [ROAD-Waymo baseline](https://github.com/salmank255/ROAD_plus_plus_Baseline), which contains a dataloader class and evaluation scripts required for all the tasks in ROAD dataset. 

You can find the **evaluation** functions in [modules/evaluation.py](https://github.com/salmank255/ROAD_plus_plus_Baseline/blob/master/modules/evaluation.py).

## Plotting annotations
![](plot.gif)
