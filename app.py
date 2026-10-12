import gradio as gr
from generate import generate_bars

iface = gr.Interface(
    fn = generate_bars,
    inputs = [
        gr.Textbox(lines = 2, placeholder='Enter a starting bar...', label = "Prompt"),
        gr.Slider(0.1, 2.0, value = 0.9, label = 'Temperature'),
        gr.Slider(1.0, 2.0, value = 1.15, label = 'Repitition Penalty')
    ],
    outputs = gr.Textbox(lines = 40, label = 'Generated lyrics'),
    title = 'Rap Lyric Generator (131M Parameters)',
    description='A custom transformer based on GPT-2 architecture, trained on 700M tokens to write multi-syllable rap verses.'
)

iface.launch(share = True)