# CS234_NeedleManipulation
The codebase for Stanford's CS234 Final Project on Needle Manipulation in Surgical Robot Environment.
It contains necessary files to conduct evaluations of multiple different environments on NeedleReach, NeedleTrack and NeedleGrasp Task.
It contains modifications to the OpenAI gym that makes it compatible with RMA framework's training process.
It is based on SurRoL, an open-source RL framework for surgical robotic learning available at https://github.com/med-air/SurRoL.

To run state-based SAC policy on needle reaching task:
```
python train.py
```

To run adaptation module training on needle reaching task:
```
python train_adapt.py
```

To run End2End Visual Policy training on needle reaching task:
First swap out the psm_env.py and the surrol_env.py with versions in the same subdirectory but with ImageObs ending.
Then execute
```
python train_end2endvisual.py
```
