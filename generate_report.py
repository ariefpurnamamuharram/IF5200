import os
import argparse
import torch

from engine.inference import Inference
from engine.report import generate_report


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

    # Generate the report
    report = generate_report(args.image)

    print('\nResults:')
    print(report, '\n')
    print('Done!')
