import argparse
import os

import gradio as gr

from src.report.report import RadiologyReport

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=7860,
                        required=False, help='Server port')
    parser.add_argument('--use_gpu', type=bool,
                        required=False, help='Use GPU device')
    args = parser.parse_args()

    # Gradio
    SAMPLES_DIR = 'sys/samples'
    radiology_report = RadiologyReport(use_gpu=args.use_gpu)
    gr \
        .Interface(fn=radiology_report.produce_report,
                   inputs=[gr.Image(type='pil')],
                   outputs='text',
                   examples=list(map(lambda x: os.path.join(SAMPLES_DIR, x), os.listdir(SAMPLES_DIR))),
                   title='Automated CXR Report Generator',
                   allow_flagging='never') \
        .launch(server_port=args.port)
