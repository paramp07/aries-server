import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import scipy.signal
import csv
import sys

class AudioInference:
    def __init__(self, target_sr=16000):
        # Set a local cache directory for TFHub to avoid corrupted global cache issues
        import os
        if "TFHUB_CACHE_DIR" not in os.environ:
            cache_dir = os.path.join(os.getcwd(), ".tfhub_cache")
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            os.environ["TFHUB_CACHE_DIR"] = cache_dir

        print("Loading YAMNet model for general human vocal detection...")
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("YAMNet Model Loaded Successfully!")
        self.target_sr = target_sr
        
        # Load class map and identify human vocalization classes
        class_map_path = self.yamnet_model.class_map_path().numpy().decode('utf-8')
        self.class_names = []
        with tf.io.gfile.GFile(class_map_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.class_names.append(row['display_name'])
                
        human_sounds = [
            "Speech", "Child speech, kid speaking", "Conversation", "Babble", "Shout", "Bellow", "Whoop", 
            "Yell", "Children shouting", "Screaming", "Whispering", "Laughter", "Baby laughter", "Giggle", 
            "Snicker", "Belly laugh", "Chuckle, chortle", "Crying, sobbing", "Baby cry, infant cry", 
            "Whimper", "Wail, moan", "Sigh", "Singing", "Choir", "Yodeling", "Chant", "Mantra", 
            "Child singing", "Synthetic singing", "Rapping", "Humming", "Groan", "Grunt", "Whistling", 
            "Breathing", "Wheeze", "Snoring", "Gasp", "Pant", "Snort", "Cough", "Throat clearing", "Sneeze", "Sniff"
        ]
        self.human_indices = [i for i, name in enumerate(self.class_names) if name in human_sounds]
        print(f"Tracking {len(self.human_indices)} human-related sound classes.")

    def run_inference(self, audio_buffer):
        """
        Expects audio_buffer as a float32 numpy array at 16000 Hz.
        Returns top_class and human_prob.
        """
        # Predict audio events with YAMNet
        scores, embeddings, spectrogram = self.yamnet_model(audio_buffer)
        
        # Extract probability (mean of frames within the buffer)
        probs = scores.numpy().mean(axis=0)
        human_prob = np.sum(probs[self.human_indices])
        
        # Find the loudest specific thing happening
        top_class = self.class_names[probs.argmax()]
        
        return top_class, float(human_prob)

def main():
    import pyaudiowpatch as pyaudio
    # Existing standalone loopback code refactored to use AudioInference class
    inference = AudioInference()
    
    p = pyaudio.PyAudio()
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        print("WASAPI not found. This script requires Windows.")
        sys.exit(1)

    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    loopback_device = None
    if not default_speakers["isLoopbackDevice"]:
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                loopback_device = loopback
                break
    else:
        loopback_device = default_speakers

    if not loopback_device:
        print("Could not find loopback device.")
        sys.exit(1)

    sample_rate = int(loopback_device["defaultSampleRate"])
    channels = loopback_device["maxInputChannels"]
    target_sr = 16000
    needs_resample = sample_rate != target_sr

    READ_CHUNK_S = 0.25
    CHUNK = int(sample_rate * READ_CHUNK_S)
    YAMNET_FRAME_SAMPLES = int(target_sr * 1.0)
    audio_buffer = np.zeros(YAMNET_FRAME_SAMPLES, dtype=np.float32)

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=loopback_device["index"])

    print("\n[ Started listening... Press Ctrl+C to stop ]\n")
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels).mean(axis=1).astype(np.int16)
            
            if needs_resample:
                num_samples_target = int(target_sr * READ_CHUNK_S)
                audio_data = scipy.signal.resample(audio_data, num_samples_target).astype(np.int16)
                
            audio_float32 = audio_data.astype(np.float32) / 32768.0
            audio_buffer = np.roll(audio_buffer, -len(audio_float32))
            audio_buffer[-len(audio_float32):] = audio_float32
            
            top_class, human_prob = inference.run_inference(audio_buffer)
            is_human_vocal = human_prob > 0.15
            
            top_class_short = (top_class[:15] + '..') if len(top_class) > 17 else top_class
            display_str = f"Top sound: {top_class_short:<17} | "
            if is_human_vocal:
                display_str += f"🗣️ HUMAN VOCAL ({human_prob:.2f})    \r"
            else:
                display_str += f"🔇 Non-vocal... ({human_prob:.2f})    \r"
            sys.stdout.write(display_str)
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()