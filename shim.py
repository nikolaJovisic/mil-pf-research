import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message=".*Reliance on distutils from stdlib is deprecated.*")
warnings.filterwarnings("ignore", message="xFormers is available.*")

REPOS_DIR = "/home/nikola.jovisic.ivi/nj"

sys.path.append(str(Path(REPOS_DIR)))
sys.path.append(str(Path(REPOS_DIR + "mammo_datasets")))
sys.path.append(str(Path(REPOS_DIR + "mammo_filter")))
sys.path.append(str(Path(REPOS_DIR + "mammo_filter/head_training")))
sys.path.append(str(Path(REPOS_DIR + "mammo_filter/embedding_inference")))


from mammo_datasets import *
from mammo_filter import *
from mammo_filter.head_training import *
from mammo_filter.embedding_inference import *
