import Augmentor
from pathlib import Path

data_folder = Path("../../muestras/aceituna_sample_da")
p = Augmentor.Pipeline(data_folder)
p.rotate(probability=0.8, max_left_rotation=0, max_right_rotation=25)
p.zoom(probability=0.8, min_factor=0.6, max_factor=1.3)
p.flip_random(probability=0.8)
p.sample(50)
