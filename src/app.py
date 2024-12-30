from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import torch
from melo.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import os
import tempfile

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Paths and configuration
ckpt_converter = 'resources/checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
reference_speaker = 'resources/voice/sample_1.mp3'  # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

# Temporary directory for audio files
temp_dir = tempfile.gettempdir()

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    data = request.json
    text = data.get('text', '')
    language = data.get('language', 'EN')
    speed = float(data.get('speed', 0.8))  # Default speed is 0.8

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    try:
        # Initialize TTS model
        model = TTS(language=language, device=device)
        speaker_ids = model.hps.data.spk2id

        # Select the speaker
        speaker_id = None
        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')
            if speaker_key in ["en-br"]:  # Adjust speaker as needed
                source_se = torch.load(f'resources/checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
                break

        if not speaker_id:
            return jsonify({'error': 'Speaker not found for the specified language'}), 400

        # Generate audio file
        src_path = os.path.join(temp_dir, 'tmp.wav')
        model.tts_to_file(text, speaker_id, src_path, speed=speed)

        # Apply tone color conversion
        save_path = os.path.join(temp_dir, 'output.wav')
        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=save_path,
            message="@MyShell"
        )

        # Return audio file or stream
        if data.get('stream', False):
            def generate():
                with open(save_path, 'rb') as f:
                    while chunk := f.read(1024):
                        yield chunk
            return Response(generate(), mimetype='audio/wav')
        else:
            return send_file(save_path, as_attachment=True, download_name='output.wav')

    except Exception as e:
        raise
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
