import argparse
import gradio as gr
from engine.report import generate_report


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int,
                        required=False, help='Server port')
    args = parser.parse_args()

    # Server port
    if args.port is not None:
        server_port = args.port
    else:
        server_port = 7860

    # Gradio
    gr.Interface(fn=generate_report,
                 inputs=gr.Image(type='pil'),
                 outputs='text',
                 examples=['sample_000.png', 'sample_001.png'],
                 title='Automated CXR Report Generator',
                 allow_flagging='never').launch(server_port=server_port)
