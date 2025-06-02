import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Brickfield dataset configuration
CLASSES = ['non-brickfield', 'brickfield']
PALETTE = np.array([
    [0, 0, 0],           # non-brickfield (black)
    [255, 128, 0],       # brickfield (orange)
], dtype=np.uint8)

def load_image(img_path):
    """Load and convert image to RGB"""
    img = Image.open(img_path).convert('RGB')
    return np.array(img)

def load_mask(mask_path, is_prediction=False):
    """Load mask - handle both GT and prediction formats"""
    if is_prediction:
        # Predictions are saved as indexed PNG with palette
        mask = Image.open(mask_path)
        if mask.mode == 'P':
            # Convert palette image to class indices
            mask = np.array(mask)
        else:
            mask = np.array(mask.convert('L'))
    else:
        # Ground truth masks
        mask = Image.open(mask_path)
        if mask.mode == 'RGB':
            # Convert RGB mask to class indices
            mask_rgb = np.array(mask)
            mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
            for i, color in enumerate(PALETTE):
                mask[np.all(mask_rgb == color, axis=2)] = i
        else:
            mask = np.array(mask.convert('L'))
    
    return mask

def mask_to_rgb(mask):
    """Convert class indices to RGB using palette"""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        rgb[mask == i] = color
    return rgb

def create_overlay(image, mask, alpha=0.5):
    """Create overlay of image and colored mask"""
    mask_rgb = mask_to_rgb(mask)
    overlay = cv2.addWeighted(image, 1-alpha, mask_rgb, alpha, 0)
    return overlay

def visualize_comparison(img_path, gt_path, pred_path, save_path=None):
    """
    Visualize image, ground truth, and prediction side by side
    
    Args:
        img_path: Path to input image
        gt_path: Path to ground truth mask
        pred_path: Path to predicted mask
        save_path: Optional path to save visualization
    """
    # Load data
    image = load_image(img_path)
    gt_mask = load_mask(gt_path, is_prediction=False)
    pred_mask = load_mask(pred_path, is_prediction=True)
    
    # Create RGB masks
    gt_rgb = mask_to_rgb(gt_mask)
    pred_rgb = mask_to_rgb(pred_mask)
    
    # Create overlays
    gt_overlay = create_overlay(image, gt_mask, alpha=0.4)
    pred_overlay = create_overlay(image, pred_mask, alpha=0.4)
    
    # Calculate accuracy
    valid_pixels = gt_mask != 255  # Exclude ignore pixels
    accuracy = np.mean(gt_mask[valid_pixels] == pred_mask[valid_pixels]) * 100
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'SemiVL Brickfield Segmentation - Accuracy: {accuracy:.1f}%', fontsize=16)
    
    # Top row - masks only
    axes[0,0].imshow(image)
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(gt_rgb)
    axes[0,1].set_title('Ground Truth')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(pred_rgb)
    axes[0,2].set_title('Prediction')
    axes[0,2].axis('off')
    
    # Bottom row - overlays
    axes[1,0].imshow(image)
    axes[1,0].set_title('Original Image')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(gt_overlay)
    axes[1,1].set_title('GT Overlay')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(pred_overlay)
    axes[1,2].set_title('Prediction Overlay')
    axes[1,2].axis('off')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=np.array(PALETTE[i])/255, 
                                   label=CLASSES[i]) for i in range(len(CLASSES))]
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    
    return accuracy

def batch_visualize(test_dir, pred_dir, save_dir=None, num_samples=5):
    """
    Visualize multiple samples from test set
    
    Args:
        test_dir: Directory containing test/images and test/gts
        pred_dir: Directory containing predictions
        save_dir: Optional directory to save visualizations
        num_samples: Number of samples to visualize
    """
    images_dir = os.path.join(test_dir, 'images')
    gts_dir = os.path.join(test_dir, 'gts')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    image_files = sorted(image_files)[:num_samples]
    
    accuracies = []
    
    for img_file in image_files:
        # Construct paths
        img_path = os.path.join(images_dir, img_file)
        gt_file = img_file.replace('.png', '_gt.png')
        gt_path = os.path.join(gts_dir, gt_file)
        pred_path = os.path.join(pred_dir, gt_file)
        
        if not all(os.path.exists(p) for p in [img_path, gt_path, pred_path]):
            print(f"Skipping {img_file} - missing files")
            continue
        
        print(f"\nVisualizing: {img_file}")
        
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f'viz_{img_file}')
        
        accuracy = visualize_comparison(img_path, gt_path, pred_path, save_path)
        accuracies.append(accuracy)
        
        print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracies:
        mean_acc = np.mean(accuracies)
        print(f"\nMean Accuracy across {len(accuracies)} samples: {mean_acc:.1f}%")
    
    return accuracies

# Example usage
if __name__ == "__main__":
    # Single image visualization
    # img_path = "/app/SemiVL_Brickfield/data/BrickField_512/test/images/patch_003_009.png" 
    # gt_path = "/app/SemiVL_Brickfield/data/BrickField_512/test/gts/patch_003_009_gt.png"
    # pred_path = "/app/SemiVL_Brickfield/output_predictions/patch_003_009_gt.png"  # Update with your prediction path
    
    # if all(os.path.exists(p) for p in [img_path, gt_path, pred_path]):
    #     visualize_comparison(img_path, gt_path, pred_path, "sample_visualization.png")
    # else:
    #     print("Update paths to your actual file locations")
    
    # Batch visualization
    batch_visualize(
        "/app/data/BrickField_512/test",
        "/app/bangladesh-brickfield-segmentation/SemiVL_Brickfield/prediction_outputs",
        "/app/bangladesh-brickfield-segmentation/SemiVL_Brickfield/visualizations",
        num_samples=100
    )

