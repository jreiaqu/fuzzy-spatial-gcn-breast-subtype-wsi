"""
Shared path configuration for all scripts in this repository.

All paths are derived from the repository root, making the codebase portable.
To use in any script:

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from _paths import *

Override any path via environment variables:
    export MOLSUB_DATA_DIR=/path/to/data
    export MOLSUB_WEIGHTS_DIR=/path/to/weights
    export MOLSUB_RESULTS_DIR=/path/to/results
"""
import os
import sys

# Repository root (one level up from scripts/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add models/ to Python path for importing model classes
sys.path.insert(0, os.path.join(REPO_ROOT, 'models'))

# ---------------------------------------------------------------------------
# Top-level directories (overridable via environment variables)
# ---------------------------------------------------------------------------
DATA_DIR    = os.environ.get('MOLSUB_DATA_DIR', os.path.join(REPO_ROOT, 'data'))
WEIGHTS_DIR = os.environ.get('MOLSUB_WEIGHTS_DIR', os.path.join(REPO_ROOT, 'weights'))
RESULTS_DIR = os.environ.get('MOLSUB_RESULTS_DIR', os.path.join(REPO_ROOT, 'results'))

# ---------------------------------------------------------------------------
# BCNB dataset paths
# ---------------------------------------------------------------------------
BCNB_GRAPHS      = os.path.join(DATA_DIR, 'BCNB', 'graphs')
BCNB_GT           = os.path.join(DATA_DIR, 'BCNB', 'ground_truth', 'patient-clinical-data.xlsx')
BCNB_SPLITS       = os.path.join(DATA_DIR, 'BCNB', 'splits')
BCNB_IMAGES       = os.path.join(DATA_DIR, 'BCNB', 'images')
BCNB_ANNOTATIONS  = os.path.join(DATA_DIR, 'BCNB', 'annotations')

# ---------------------------------------------------------------------------
# SBC dataset paths
# ---------------------------------------------------------------------------
SBC_GRAPHS  = os.path.join(DATA_DIR, 'SBC', 'graphs')
SBC_CONCH   = os.path.join(DATA_DIR, 'SBC', 'graphs_CONCH')
SBC_GT      = os.path.join(DATA_DIR, 'SBC', 'ground_truth', 'ground_truth.xlsx')
SBC_FOLDS   = os.path.join(DATA_DIR, 'SBC', 'folds')

# ---------------------------------------------------------------------------
# Model weights
# ---------------------------------------------------------------------------
GCN_WEIGHTS          = os.path.join(WEIGHTS_DIR, 'pretrained')
GCN_WEIGHTS_ORIGINAL = os.path.join(WEIGHTS_DIR, 'pretrained')
GCN_WEIGHTS_RETRAINED = os.path.join(WEIGHTS_DIR, 'retrained')

# ---------------------------------------------------------------------------
# Legacy aliases (backward compatibility)
# ---------------------------------------------------------------------------
MOLSUB_ROOT = REPO_ROOT
MOLSUB = REPO_ROOT
CODE_DIR = os.path.join(REPO_ROOT, 'models')
MOLSUB_CODE = os.path.join(REPO_ROOT, 'models')
RESULTS = RESULTS_DIR
