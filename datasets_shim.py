import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message=".*Reliance on distutils from stdlib is deprecated.*")
warnings.filterwarnings("ignore", message="xFormers is available.*")

repos_dir = "/home/nikola.jovisic.ivi/nj/"

sys.path.append(str(Path(repos_dir)))
sys.path.append(str(Path(repos_dir + "mammo_datasets")))
sys.path.append(str(Path(repos_dir + "mammo_filter/head_training")))
sys.path.append(str(Path(repos_dir + "mammo_filter/embedding_inference")))

from mammo_datasets import *
from mammo_filter.head_training import *