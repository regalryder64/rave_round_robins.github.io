import os
import shutil
import torch
import torchaudio

# Setup your paths
model_path = "runs/surface_run_6d48ff3188/version_0/checkpoints/SundayMorning2.ts"
input_audio_path = "input_samples/Snare Sample 1.wav"

# 1. Extract the name for the folder (e.g., "Snare Sample 1")
sample_name = os.path.splitext(os.path.basename(input_audio_path))[0]
output_dir = os.path.join("output_samples", sample_name)

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Copy the original file into the new folder so they are kept together
original_copy_path = os.path.join(output_dir, f"ORIGINAL_{os.path.basename(input_audio_path)}")
shutil.copy(input_audio_path, original_copy_path)

# 2. Load your exported RAVE model
print(f"Loading model: {model_path}")
model = torch.jit.load(model_path)

# 3. Load your input snare drum
print(f"Loading audio: {input_audio_path}")
wav, sr = torchaudio.load(input_audio_path)

# RAVE expects mono audio with a batch dimension: shape (1, 1, samples)
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)
wav = wav.unsqueeze(0) 

# 4. Encode the snare into the latent space
print(f"Generating 20 variations in: {output_dir}/")
with torch.no_grad():
    z = model.encode(wav)

variance = 3
# 5. Generate 20 variations
for i in range(20):
    # Inject a tiny amount of random Gaussian noise into the DNA.
    # The '0.3' is your variance multiplier. 
    noisy_z = z + torch.randn_like(z) * variance
    
    # Decode the mutated DNA back into audio
    with torch.no_grad():
        output = model.decode(noisy_z)
    
    # Construct the filename and path
    filename = f"{sample_name}_variation_{i+1}.wav"
    filepath = os.path.join(output_dir, filename)
    
    # Remove batch dimension to save
    out_wav = output.squeeze(0)
    torchaudio.save(filepath, out_wav, sr)
    
    print(f"Exported: {filename}")

print("\nDone! All files saved successfully.")