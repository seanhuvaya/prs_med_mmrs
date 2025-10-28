"""
Modality configuration for PRS-Med datasets.
Maps dataset folders to their corresponding imaging modalities.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModalityMapping:
    """Configuration for mapping dataset folders to modalities."""
    dataset_name: str
    modality_type: str
    description: str
    image_type: str
    expected_channels: int = 3
    typical_size: tuple = (224, 224)

# Define the mapping from your dataset folders to standard modalities
MODALITY_MAPPINGS = {
    "brain_tumors_ct_scan": ModalityMapping(
        dataset_name="brain_tumors_ct_scan",
        modality_type="CT",
        description="Brain Tumors CT Scan",
        image_type="CT Scan",
        expected_channels=3
    ),
    "breast_tumors_ct_scan": ModalityMapping(
        dataset_name="breast_tumors_ct_scan", 
        modality_type="CT",
        description="Breast Tumors CT Scan",
        image_type="CT Scan",
        expected_channels=3
    ),
    "dental_xray": ModalityMapping(
        dataset_name="dental_xray",
        modality_type="X-ray",
        description="Dental X-ray",
        image_type="X-ray",
        expected_channels=1  # X-rays are typically grayscale
    ),
    "lung_CT": ModalityMapping(
        dataset_name="lung_CT",
        modality_type="CT", 
        description="Lung CT Scan",
        image_type="CT Scan",
        expected_channels=3
    ),
    "lung_Xray": ModalityMapping(
        dataset_name="lung_Xray",
        modality_type="X-ray",
        description="Lung X-ray",
        image_type="X-ray", 
        expected_channels=1  # X-rays are typically grayscale
    ),
    "polyp_endoscopy": ModalityMapping(
        dataset_name="polyp_endoscopy",
        modality_type="Endoscopy",
        description="Polyp Endoscopy",
        image_type="Endoscopy",
        expected_channels=3
    ),
    "skin_rgbimage": ModalityMapping(
        dataset_name="skin_rgbimage",
        modality_type="RGB Image",
        description="Skin RGB Image",
        image_type="RGB Image",
        expected_channels=3
    )
}

# Standard modality categories
STANDARD_MODALITIES = {
    "CT": {
        "name": "CT Scan",
        "description": "Computed Tomography",
        "typical_use": "Internal organ imaging, tumor detection",
        "color_space": "RGB/Grayscale"
    },
    "MRI": {
        "name": "Magnetic Resonance Imaging", 
        "description": "MRI Scan",
        "typical_use": "Soft tissue imaging, brain scans",
        "color_space": "Grayscale/RGB"
    },
    "X-ray": {
        "name": "X-ray Imaging",
        "description": "X-ray Radiograph", 
        "typical_use": "Bone imaging, chest X-rays",
        "color_space": "Grayscale"
    },
    "Ultrasound": {
        "name": "Ultrasound",
        "description": "Ultrasonography",
        "typical_use": "Real-time imaging, pregnancy, heart",
        "color_space": "Grayscale/RGB"
    },
    "Endoscopy": {
        "name": "Endoscopy",
        "description": "Endoscopic Imaging",
        "typical_use": "Internal cavity examination",
        "color_space": "RGB"
    },
    "RGB Image": {
        "name": "RGB Image",
        "description": "Standard RGB Photography",
        "typical_use": "Surface imaging, dermatology",
        "color_space": "RGB"
    }
}

def get_modality_mapping(dataset_name: str) -> Optional[ModalityMapping]:
    """Get modality mapping for a dataset."""
    return MODALITY_MAPPINGS.get(dataset_name)

def get_all_modalities() -> List[str]:
    """Get list of all available modalities."""
    return list(STANDARD_MODALITIES.keys())

def get_datasets_by_modality(modality_type: str) -> List[str]:
    """Get all datasets that belong to a specific modality."""
    return [name for name, mapping in MODALITY_MAPPINGS.items() 
            if mapping.modality_type == modality_type]

def get_modality_stats() -> Dict[str, int]:
    """Get statistics about dataset distribution by modality."""
    stats = {}
    for mapping in MODALITY_MAPPINGS.values():
        modality = mapping.modality_type
        stats[modality] = stats.get(modality, 0) + 1
    return stats

def validate_modality_mapping(dataset_name: str) -> bool:
    """Validate if a dataset has a valid modality mapping."""
    return dataset_name in MODALITY_MAPPINGS

# Print current configuration
if __name__ == "__main__":
    print("=== PRS-Med Modality Configuration ===")
    print(f"Total datasets: {len(MODALITY_MAPPINGS)}")
    print(f"Standard modalities: {list(STANDARD_MODALITIES.keys())}")
    print("\nDataset to Modality Mapping:")
    
    for dataset, mapping in MODALITY_MAPPINGS.items():
        print(f"  {dataset} → {mapping.modality_type} ({mapping.description})")
    
    print(f"\nModality Statistics:")
    stats = get_modality_stats()
    for modality, count in stats.items():
        print(f"  {modality}: {count} datasets")
    
    print(f"\nDatasets by Modality:")
    for modality in STANDARD_MODALITIES.keys():
        datasets = get_datasets_by_modality(modality)
        if datasets:
            print(f"  {modality}: {', '.join(datasets)}")
        else:
            print(f"  {modality}: No datasets available")
