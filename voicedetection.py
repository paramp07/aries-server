from nemo.collections.asr.models import EncDecClassificationModel

model = EncDecClassificationModel.from_pretrained(
    model_name="nvidia/voiceactivitydetection_marblenet"
)

pred = model.transcribe(["audio.wav"])

print(pred)  # Outputs speech/non-speech probabilities