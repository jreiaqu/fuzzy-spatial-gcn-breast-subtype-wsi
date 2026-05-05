import matplotlib.pyplot as plt

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def create_confusion_matrix(cm_data, labels, ax, cmap='Blues'):
    """
    Create a compact, clean confusion matrix following supervisor feedback
    - Much smaller overall size
    - Larger numbers and fonts
    - No scientific notation, only integers
    - Minimal unnecessary elements
    """
    
    # Create the heatmap with clean styling
    im = ax.imshow(cm_data, interpolation='nearest', cmap=cmap, aspect='equal')
    
    # Set tick marks and labels with larger fonts
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    
    # Smaller, cleaner labels
    if len(labels) == 4:  # 4-class case - use abbreviated labels
        short_labels = ['L-A', 'L-B', 'HER2+', 'TNBC']
        ax.set_xticklabels(short_labels, fontsize=14, fontweight='bold')
        ax.set_yticklabels(short_labels, fontsize=14, fontweight='bold')
    else:
        ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
        ax.set_yticklabels(labels, fontsize=14, fontweight='bold')
    
    # Add LARGE number annotations - this is the key improvement
    thresh = cm_data.max() / 2.
    for i in range(cm_data.shape[0]):
        for j in range(cm_data.shape[1]):
            # Large integer formatting - no scientific notation
            text = f'{int(cm_data[i, j])}'
            color = "white" if cm_data[i, j] > thresh else "black"
            ax.text(j, i, text,
                   ha="center", va="center",
                   color=color,
                   fontsize=18, fontweight='bold')  # Much larger font
    
    # Remove unnecessary elements for cleaner look
    ax.set_xticks(tick_marks, minor=False)
    ax.set_yticks(tick_marks, minor=False)
    
    # No colorbar - cleaner appearance
    
    return ax

# Define the confusion matrix data from your Figure 2
# You'll need to replace these values with your actual data

# NCA Results (Top row in your figure)
nca_2clf = np.array([[171, 21],    # Adjust these values to match your actual results
                     [16, 10]])

nca_3clf = np.array([[81, 45, 7],   # Adjust these values to match your actual results
                     [10, 45, 4],
                     [4, 13, 9]])

nca_4clf = np.array([[26, 25, 10, 4],   # Adjust these values to match your actual results
                     [20, 28, 15, 5],
                     [9, 10, 23, 17],
                     [5, 3, 6, 12]])

# CA Results (Bottom row in your figure) 
ca_2clf = np.array([[179, 13],    # Adjust these values to match your actual results
                    [12, 14]])

ca_3clf = np.array([[90, 31, 12],   # Adjust these values to match your actual results
                    [10, 38, 11],
                    [3, 9, 14]])

ca_4clf = np.array([[36, 14, 13, 2],   # Adjust these values to match your actual results
                    [14, 32, 17, 5],
                    [4, 13, 34, 8],
                    [2, 3, 10, 11]])

# Define labels for each classification task
labels_2clf = ['Other', 'TNBC']
labels_3clf = ['Luminals', 'HER2+', 'TNBC']
labels_4clf = ['Luminal A', 'Luminal B', 'HER2+', 'TNBC']

# Create individual matrices as separate files for LaTeX integration
data_list = [nca_2clf, nca_3clf, nca_4clf, ca_2clf, ca_3clf, ca_4clf]
labels_list = [labels_2clf, labels_3clf, labels_4clf, labels_2clf, labels_3clf, labels_4clf]
filenames = [
    'OTHERvsTNBC_BB_NCA_10x_BCNB_test_cfsn_matrix_best_f1_train_end.png',
    'LUMINALSvsHER2vsTNBC_BB_NCA_10x_BCNB_test_cfsn_matrix_best_f1_train_end.png',
    'LUMINALAvsLAUMINALBvsHER2vsTNBC_BB_NCA_10x_BCNB_test_cfsn_matrix_best_f1_train_end.png',
    'OTHERvsTNBC_BB_CA_10x_BCNB_test_cfsn_matrix_best_f1_train_end.png',
    'LUMINALSvsHER2vsTNBC_BB_CA_10x_BCNB_test_cfsn_matrix_best_f1_train_end.png',
    'LUMINALAvsLAUMINALBvsHER2vsTNBC_BB_CA_10x_BCNB_test_cfsn_matrix_best_f1_train_end.png'
]

# Create each matrix as individual compact files
for idx, (data, labels, filename) in enumerate(zip(data_list, labels_list, filenames)):
    # Determine figure size based on matrix size - much smaller and more compact
    if len(labels) == 2:
        figsize = (3, 3)  # Very compact for 2x2
    elif len(labels) == 3:
        figsize = (3.5, 3.5)  # Compact for 3x3
    else:
        figsize = (4, 4)  # Still compact for 4x4
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    create_confusion_matrix(data, labels, ax)
    
    # Remove all excess whitespace - very tight
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    # Save with exact same filename structure you use
    plt.savefig(f'Figures/cfs/{filename}', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()  # Close to save memory
    
    print(f"Created: {filename}")

# Also create the combined version if needed
fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # Much smaller combined figure

for idx, (data, labels) in enumerate(zip(data_list, labels_list)):
    row = idx // 3
    col = idx % 3
    create_confusion_matrix(data, labels, axes[row, col])

plt.tight_layout()
plt.subplots_adjust(hspace=0.15, wspace=0.15)

# Save the figure
plt.savefig('confusion_matrices_improved.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('confusion_matrices_improved.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

plt.show()

print("Confusion matrices created successfully!")
print("Files saved as: confusion_matrices_improved.pdf and confusion_matrices_improved.png")
print("\nTo use this script:")
print("1. Replace the dummy values in nca_2clf, nca_3clf, etc. with your actual confusion matrix data")
print("2. Adjust colors by changing the 'cmap' parameter if needed")
print("3. Modify font sizes or layout parameters as required")
print("4. Run the script to generate publication-ready confusion matrices")