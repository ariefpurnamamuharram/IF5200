from PIL import Image
from torchvision import transforms as T


class ToTensorTransform:

    def __init__(self):

        super(ToTensorTransform, self).__init__()

        # Image transformations
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_transformer(self):

        return self.transform

    def set_transformer(self, transform):

        self.transform = transform

    def transform(image: Image):

        # Transform the image
        img = self.transform(image)

        # Return the image
        return img


def read_image(image_path: str) -> Image:

    # Load image
    im = Image.open(image_path)

    # Convert to RGB
    im = im.convert('RGB')

    # Return loaded image
    return im


def get_square_resize(im: Image, dim: int) -> Image:

    # Resize the image
    im = im.resize((dim, dim))

    # Return the resized image
    return im


def get_segment(im: Image, segment: int) -> Image:

    if segment == 0 or segment > 3:
        raise ValueError('Segment number is out of index!')

    # Crop the image
    if segment == 1:
        im = im.crop((0, 0, im.width, round((im.height * 0.5), 0)))
    elif segment == 2:
        im = im.crop((0, round((im.height * 0.5), 0), im.width, im.height))
    elif segment == 3:
        im = im.crop((0, round((im.height * 0.15), 0), im.width,
                     (im.height - round((im.height * 0.15), 0))))

    # Return the image segment
    return im
