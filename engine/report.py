import os
import torch
from PIL import Image

from engine.inference import Inference
from engine.utils.image import read_image, get_segment, ToTensorTransform


def build_report(cardiomegaly: int = 0, effusion: int = 0) -> str:

    try:
        report_text = f'Pada foto radiologi dada yang diterima diperoleh temuan-temuan sebagai berikut: {"Bentuk jantung tampak baik, tidak ditemukan tanda-tanda kardiomegali" if bool(cardiomegaly) else "Terdapat gambaran kardiomegali, CTR > 50%"}. {"Tidak tampak gambaran efusi" if bool(effusion) else "Tampak gambaran efusi"} pada lapang paru.'
    except BaseException:
        raise ValueError('Report error!')

    return report_text


def generate_report(
        img: (str or Image),
        device: torch.device = torch.device('cpu')) -> str:

    # Read image
    if isinstance(img, str):
        img = read_image(img)
    img_segment1 = get_segment(img, 1)
    img_segment2 = get_segment(img, 2)
    img_segment3 = get_segment(img, 3)

    # Transform the images into tensors
    transformer = ToTensorTransform()
    img_segment1 = transformer.transform(img_segment1)
    img_segment2 = transformer.transform(img_segment2)
    img_segment3 = transformer.transform(img_segment3)

    # Load the models
    MODELS_DIR = 'enabled/models'
    model_cardiomegaly = torch.load(
        os.path.join(
            MODELS_DIR,
            'model_cardiomegaly.pth'))
    model_effusion = torch.load(os.path.join(MODELS_DIR, 'model_effusion.pth'))

    # Make inferences
    inference = Inference(device)
    result_cardiomegaly = inference.make_prediction(
        model_cardiomegaly, img_segment2.unsqueeze(0)).item()
    result_effusion = inference.make_prediction(
        model_effusion, img_segment2.unsqueeze(0)).item()

    # Generate the report
    report = build_report(result_cardiomegaly, result_effusion)

    # Return the report
    return report
