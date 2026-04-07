import numpy as np
from app.ml.audio.yamnet import AudioInference

def test_inference():
    print("Testing AudioInference with dummy data...")
    inference = AudioInference()
    
    # 1. Test with silence
    silence = np.zeros(16000, dtype=np.float32)
    top_class, human_prob = inference.run_inference(silence)
    print(f"Silence -> Top Sound: {top_class}, Human Vocal: {human_prob:.4f}")
    
    # 2. Test with random noise (to ensure it doesn't crash)
    noise = np.random.uniform(-1.0, 1.0, 16000).astype(np.float32)
    top_class, human_prob = inference.run_inference(noise)
    print(f"Noise -> Top Sound: {top_class}, Human Vocal: {human_prob:.4f}")

    print("Verification completed.")

if __name__ == "__main__":
    test_inference()
