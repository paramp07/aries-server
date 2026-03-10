import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pyaudiowpatch as pyaudio
import scipy.signal
import sys
import csv

def main():
    # Load Google's YAMNet Model for Audio Event Detection
    print("Loading YAMNet model for general human vocal detection...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    # Load class map and identify human vocalization classes
    class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
            
    human_sounds = [
        "Speech", "Child speech, kid speaking", "Conversation", "Babble", "Shout", "Bellow", "Whoop", 
        "Yell", "Children shouting", "Screaming", "Whispering", "Laughter", "Baby laughter", "Giggle", 
        "Snicker", "Belly laugh", "Chuckle, chortle", "Crying, sobbing", "Baby cry, infant cry", 
        "Whimper", "Wail, moan", "Sigh", "Singing", "Choir", "Yodeling", "Chant", "Mantra", 
        "Child singing", "Synthetic singing", "Rapping", "Humming", "Groan", "Grunt", "Whistling", 
        "Breathing", "Wheeze", "Snoring", "Gasp", "Pant", "Snort", "Cough", "Throat clearing", "Sneeze", "Sniff"
    ]
    human_indices = [i for i, name in enumerate(class_names) if name in human_sounds]
    print(f"Tracking {len(human_indices)} human-related sound classes (screaming, crying, laughing, grunting, etc.).")

    # Initialize PyAudio with the WASAPI patch for desktop loopback
    p = pyaudio.PyAudio()
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        print("WASAPI not found. This script requires Windows.")
        sys.exit(1)

    # Find the default speaker loopback device
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
        print("Could not find a loopback device for your default speakers.")
        sys.exit(1)

    # Get native sample rate and channels of the desktop audio
    sample_rate = int(loopback_device["defaultSampleRate"])
    channels = loopback_device["maxInputChannels"]
    
    # YAMNet operates strictly at 16000 Hz mono
    target_sr = 16000
    if sample_rate == 16000:
        needs_resample = False
    else:
        needs_resample = True

    print(f"Listening on: {loopback_device['name']}")
    print(f"Native capture: {sample_rate}Hz, {channels} channels")
    print(f"VAD working at: {target_sr}Hz")

    # We want to evaluate roughly 1 second of audio context at a time for YAMNet
    # but still update quickly. So we'll read 0.25s chunks from pyaudio and maintain a 1-second rolling buffer.
    READ_CHUNK_S = 0.25
    CHUNK = int(sample_rate * READ_CHUNK_S)
    
    YAMNET_FRAME_SAMPLES = int(target_sr * 1.0) # 16000 samples = 1 second
    audio_buffer = np.zeros(YAMNET_FRAME_SAMPLES, dtype=np.float32)

    # Open the WASAPI Loopback stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=loopback_device["index"])

    print("\n[ Started listening... (Make sure audio is playing!) Press Ctrl+C to stop ]\n")
    
    # Note on WASAPI Loopback on Windows:
    # If there is absolutely zero audio playing system-wide, stream.read() will block.
    # It will start loop processing continuously as soon as any audio plays.

    try:
        while True:
            # exception_on_overflow=False so it doesn't crash if we are slightly slow
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert raw bytes to integer numpy array
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Convert stere/surround to mono by averaging channels
            if channels > 1:
                audio_data = audio_data.reshape(-1, channels).mean(axis=1).astype(np.int16)
            
            # Resample integer array if needed (e.g. 44100 -> 16000)
            if needs_resample:
                # Calculate exact chunk size target for the resampled 0.25s chunk
                num_samples_target = int(target_sr * READ_CHUNK_S)
                audio_data = scipy.signal.resample(audio_data, num_samples_target).astype(np.int16)
                
            # Convert integer array to float32 tensor
            audio_float32 = audio_data.astype(np.float32) / 32768.0
            
            # Roll buffer: shift left and append new float32 data
            audio_buffer = np.roll(audio_buffer, -len(audio_float32))
            audio_buffer[-len(audio_float32):] = audio_float32
            
            # Predict audio events with YAMNet over the 1 second context
            scores, embeddings, spectrogram = yamnet_model(audio_buffer)
            
            # Extract probability (mean of frames within the 1-second burst)
            probs = scores.numpy().mean(axis=0)
            human_prob = np.sum(probs[human_indices])
            
            # Use 0.15 threshold for general human vocalizations (YAMNet probabilities are distributed across 521 classes)
            is_human_vocal = human_prob > 0.15
            
            # Find the loudest specific thing happening
            top_class = class_names[probs.argmax()]
            top_class_short = (top_class[:15] + '..') if len(top_class) > 17 else top_class
            
            # Build display string
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