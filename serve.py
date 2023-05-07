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


    # Get image samples
    def get_image_samples() -> list:
        samples_dir = 'sys/samples'
        items = os.listdir(samples_dir)
        images = []
        for item in items:
            if os.path.splitext(item)[1] in ['.jpg', '.jpeg', '.png']:
                images.append(os.path.join(samples_dir, item))
            else:
                continue
        return images


    # Gradio
    radiology_report = RadiologyReport(use_gpu=args.use_gpu)
    gr \
        .Interface(fn=radiology_report.produce_report,
                   inputs=[gr.Image(type='pil')],
                   outputs='text',
                   examples=get_image_samples(),
                   title='Automated CXR Report Generator',
                   allow_flagging='never') \
        .launch(server_port=args.port)
