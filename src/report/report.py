import os

import torch
from PIL import Image

from src.report.utils.inference import InferenceUtils
from src.utils.image import read_image, get_segment, ToTensorTransform


class RadiologyReport:

    def __init__(self, use_gpu: bool = False):
        super(RadiologyReport, self).__init__()
        self.use_gpu = use_gpu

    @staticmethod
    def __build_report_text(cardiomegaly: int = 0, effusion: int = 0) -> str:
        """
        Args:
            cardiomegaly: Presence of cardiomegaly finding
            effusion: Presence of effusion finding

        Returns:
            Radiology report text
        """

        try:
            report_text = f'Pada foto radiologi dada yang diterima diperoleh temuan-temuan sebagai berikut: ' \
                          f'{"Bentuk jantung tampak baik, tidak ditemukan tanda-tanda kardiomegali" if bool(cardiomegaly) else "Terdapat gambaran kardiomegali, CTR > 50%"}. ' \
                          f'{"Tidak tampak gambaran efusi" if bool(effusion) else "Tampak gambaran efusi"} pada lapang paru.'
        except BaseException:
            raise ValueError('Report error!')

        return report_text

    def produce_report(self, img: (str or Image)) -> str:
        """
        Args:
            img: Radiology image, can be either the image object or the image path

        Returns:
            Radiology report text
        """

        # Read image
        if isinstance(img, str):
            img = read_image(img)

        # Slice the image
        img_segment1 = get_segment(img, 1)
        img_segment2 = get_segment(img, 2)
        img_segment3 = get_segment(img, 3)

        # Transform the sliced images into tensors
        transformer = ToTensorTransform()
        img_segment1 = transformer.transform(img_segment1)
        img_segment2 = transformer.transform(img_segment2)
        img_segment3 = transformer.transform(img_segment3)

        # Setup torch device
        device = torch.device('cuda') if self.use_gpu else torch.device('cpu')

        # Load the outputs
        models_dir = 'system/models'
        model_cardiomegaly = torch.load(os.path.join(models_dir, 'model_cardiomegaly.pth'), map_location=device)
        model_effusion = torch.load(os.path.join(models_dir, 'model_effusion.pth'), map_location=device)

        # Make inferences
        inference = InferenceUtils(device=device)
        result_cardiomegaly = inference \
            .make_prediction(model_cardiomegaly, img_segment2.unsqueeze(0)) \
            .item()
        result_effusion = inference \
            .make_prediction(model_effusion, img_segment2.unsqueeze(0)) \
            .item()

        # Generate the report
        report = self.__build_report_text(result_cardiomegaly, result_effusion)

        # Return the report
        return report
