# Download and pre-process ROAD++ dataset

Here we provide the **download** and **pre-processing instructions** for the ROAD++ dataset, that is released through ROAD++ challenge: [The Second Workshop & Challenge on Event Detection for Situation Awareness in Autonomous Driving](https://sites.google.com/view/road-plus-plus/home) and uses [3D-RetinaNet](https://github.com/salmank255/ROAD_plus_plus_Baseline) code as a **baseline**, which also contains the evaluation code. 

## Main Features

- Action annotations for human as well as other road agents, e.g. Turning-right, Moving-away etc. 
- Agent type labels, e.g. Pedestrian, Car, Cyclist, Large-Vehicle, Emergency-Vehicle etc.
- Semantic location labels of the location of agent, e.g. in vehicle lane, in right pavement etc.
- 198K frames from 1000 videos annotated, each video is 20 seconds long on an average.
- track/tube id annotated for every bounding box on every frame for every agent in the scene.
- 54K tubes/tracks of individual agents.
- 3.9M bounding  box-level agent  labels.
- 4.3M and 4.2M bounding box-level action and location labels.

## Attribution
ROAD++ dataset is build upon [WAYMO Open Dataset](https://waymo.com/open/). If you find the original dataset useful in your work, please cite it using the citation that can be found [here](https://arxiv.org/pdf/1912.04838.pdf). 

Similar to the original dataset [WAMO](https://waymo.com/privacy/), the ROAD++ dataset is licensed under a [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms/) “Non-commercial Purposes" means research, teaching, scientific publication and personal experimentation. Non-commercial Purposes include use of the Dataset to perform benchmarking for purposes of academic or applied research publication. Non-commercial Purposes does not include purposes primarily intended for or directed towards commercial advantage or monetary compensation, or purposes intended for or directed towards litigation, licensing, or enforcement, even in part. If you are interested in using the dataset for commercial purposes, please contact original creator [WAYMO Open Dataset](https://waymo.com/open/) for video content and [Fabio](https://cms.brookes.ac.uk/staff/FabioCuzzolin/) and [Salman](https://sites.google.com/view/salman-k) for event annotations.

