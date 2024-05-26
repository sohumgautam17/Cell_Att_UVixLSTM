import albumentations as A
from albumentations.pytorch import ToTensorV2



def geometric_transform():
    return A.Compose([
        A.resize(1000, 1000),
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1)
    ])

def apply_aug(image, mask):
    try:
        transform = geometric_transform()
        transformed = transform(image=image, mask=mask)

        transformed_image = transformed["image"]
        transformed_mask = transformed["mask"]

        return transformed_image, transformed_mask
    except Exception as e:
        print("Error")
        raise