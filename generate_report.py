import os
import argparse
import torch

from engine.inference import Inference
from engine.report import make_report
from engine.utils.image import read_image, get_segment, ToTensorTransform


if __name__ == '__main__':
    
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--device', type=str, required=False)
    args = parser.parse_args()
    
    # Device
    if args.device is not None:
        if str(args.device)[0:3].lower() not in 'cuda':
            raise ValueError('Device is not recognized!')
        else:
            device = torch.device(args.device)
    else:
        device = torch.device('cpu')
    
    # Read image
    img = read_image(args.image)
    img_segment1 = get_segment(img, 1)
    img_segment2 = get_segment(img, 2)
    img_segment3 = get_segment(img, 3)
    transformer = ToTensorTransform()
    img_segment1 = transformer.transform(img_segment1)
    img_segment2 = transformer.transform(img_segment2)
    img_segment3 = transformer.transform(img_segment3)
    
    # Load the models
    MODELS_DIR = 'enabled/models'
    cardiomegaly = torch.load(os.path.join(MODELS_DIR, 'model_cardiomegaly.pth'))
    effusion = torch.load(os.path.join(MODELS_DIR, 'model_effusion.pth'))
    
    # Make inferences
    inference = Inference(device)
    result_cardiomegaly = inference.make_prediction(cardiomegaly, img_segment2.unsqueeze(0)).item()
    result_effusion = inference.make_prediction(effusion, img_segment2.unsqueeze(0)).item()
    
    # Generate the report
    report = make_report(result_cardiomegaly, result_effusion)
    
    print('\nResults:')
    print(report, '\n')
    print('Done!')
       
