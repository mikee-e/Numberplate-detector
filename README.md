# Numberplate Detector
## Installation
- To manage environments it is recommended to use anaconda https://docs.anaconda.com/anaconda/install/
- After installing anaconda create virtual environment via this command
```
conda create -n npdet python=3.10
conda activate npdet
```
- Clone repository
```
git clone https://github.com/mikee-e/Numberplate-detector.git && cd Numberplate-detector
```
- Install requirements
```
pip install -r requirements.txt
cd ByteTrack
pip install requirements.txt && python setup.py develop
cd ..
```
- Run detector(replate <video_file> with your video path)
```
python numberplate_detector.py <video_file>
```
