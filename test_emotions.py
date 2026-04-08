#!/usr/bin/env python3
"""
Video Emotion Detection using JointCAM Model

This script extracts emotions (valence and arousal) from video files
using the pre-trained JointCAM model.

Usage:
    python test_emotions.py <video_path> [--model_path <model.pt>]
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================================
# Model Architecture Components
# ============================================================================

class BiLSTMExtractor(nn.Module):
    """Bidirectional LSTM for temporal feature extraction."""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.linear = nn.Linear(input_dim, hidden_dim * 2)
        self.rnn = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        # Learnable initial hidden state
        self.init_hidden = nn.Parameter(torch.zeros(num_layers * 2, hidden_dim))
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.linear(x)
        
        # Initialize hidden states
        h0 = self.init_hidden.unsqueeze(1).repeat(1, batch_size, 1)
        c0 = torch.zeros_like(h0)
        
        output, _ = self.rnn(x, (h0, c0))
        return output


class CrossModalAttention(nn.Module):
    """Cross-modal attention between audio and video."""
    
    def __init__(self, dim=512):
        super().__init__()
        self.affine_audio = nn.Linear(dim, dim)
        self.affine_video = nn.Linear(dim, dim)
        self.affine_out = nn.Linear(dim, dim)
        
    def forward(self, audio, video):
        # Compute attention weights
        audio_proj = self.affine_audio(audio)
        video_proj = self.affine_video(video)
        
        attn = torch.bmm(audio_proj, video_proj.transpose(1, 2))
        attn = F.softmax(attn / np.sqrt(audio_proj.size(-1)), dim=-1)
        
        attended = torch.bmm(attn, video)
        output = self.affine_out(attended)
        return output


class DenseCoAttention(nn.Module):
    """Dense co-attention layer."""
    
    def __init__(self, dim=1024):
        super().__init__()
        self.query_linear = nn.Linear(dim, dim)
        self.key1_linear = nn.Linear(16, 16)
        self.key2_linear = nn.Linear(16, 16)
        self.value1_linear = nn.Linear(512, 512)
        self.value2_linear = nn.Linear(512, 512)
        
    def forward(self, x1, x2):
        return x1, x2  # Simplified pass-through for inference


class DCNLayer(nn.Module):
    """Dense Co-attention Network layer."""
    
    def __init__(self, dim=512):
        super().__init__()
        self.dense_coattn = DenseCoAttention(dim * 2)
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 2, dim)),
            nn.Sequential(nn.Linear(dim * 2, dim))
        ])
        
    def forward(self, x1, x2):
        # Concatenate and process
        combined = torch.cat([x1, x2], dim=-1)
        out1 = self.linears[0](combined)
        out2 = self.linears[1](combined)
        return out1, out2


class VideoAttention(nn.Module):
    """Video attention with audio guidance."""
    
    def __init__(self, dim=512):
        super().__init__()
        self.attn = CrossModalAttention(dim)
        
    def forward(self, video, audio):
        return self.attn(video, audio)


class AudioEmbedding(nn.Module):
    """
    Learned embedding for log-mel spectrogram features.
    Projects (T, 128) log-mel features to (T, 1024) using CNN/MLP.
    """
    
    def __init__(self, input_dim=128, output_dim=1024):
        super().__init__()
        # Small CNN for temporal feature extraction (preserves temporal dimension)
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # Point-wise convolution to project to output dimension
        self.proj = nn.Conv1d(512, output_dim, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 128) log-mel features
        Returns:
            out: (batch, seq_len, 1024) embedded features
        """
        # x: (B, T, 128) -> (B, 128, T) for conv1d
        x = x.transpose(1, 2)
        
        # Apply CNN layers (preserves temporal dimension)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Project to output dimension
        x = self.proj(x)
        
        # Transpose back: (B, 1024, T) -> (B, T, 1024)
        x = x.transpose(1, 2)
        
        return x


class JointCAM(nn.Module):
    """
    JointCAM: Joint Cross-Modal Attention Model for Emotion Recognition.
    
    Processes video and audio features through BiLSTM extractors,
    applies cross-modal attention, and predicts valence/arousal.
    """
    
    def __init__(self, use_audio_embedding=False):
        super().__init__()
        
        # Audio embedding (optional, for new preprocessing)
        self.use_audio_embedding = use_audio_embedding
        if use_audio_embedding:
            self.audio_embedding = AudioEmbedding(input_dim=128, output_dim=1024)
        
        # Feature extractors (BiLSTM)
        self.video_extract = BiLSTMExtractor(input_dim=512, hidden_dim=256)
        self.audio_extract = BiLSTMExtractor(input_dim=1024, hidden_dim=256)
        
        # Attention modules
        self.video_attn = VideoAttention(dim=512)
        self.coattn = nn.ModuleDict({
            'dcn_layers': nn.ModuleList([DCNLayer(dim=512)])
        })
        
        # Joint fusion
        self.Joint = BiLSTMExtractor(input_dim=1536, hidden_dim=256)
        
        # Regressors for valence and arousal
        self.vregressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        self.aregressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
    def forward(self, video_features, audio_features):
        """
        Args:
            video_features: (batch, seq_len, 512)
            audio_features: (batch, seq_len, 1024) or (batch, seq_len, 128) if using new preprocessing
            
        Returns:
            valence: (batch, seq_len, 1) in range [-1, 1]
            arousal: (batch, seq_len, 1) in range [-1, 1]
        """
        # Handle audio features: apply embedding if enabled, or repeat if needed for backward compatibility
        if audio_features.shape[-1] == 128:
            if self.use_audio_embedding:
                # Use learned embedding to project 128 -> 1024
                audio_features = self.audio_embedding(audio_features)
            else:
                # Fallback: repeat 8x for backward compatibility (not recommended)
                audio_features = audio_features.repeat(1, 1, 8)  # (B, T, 128) -> (B, T, 1024)
        
        # Extract temporal features
        video_out = self.video_extract(video_features)  # (B, T, 512)
        audio_out = self.audio_extract(audio_features)  # (B, T, 512)
        
        # Cross-modal attention
        video_attended = self.video_attn(video_out, audio_out)
        
        # Co-attention
        video_co, audio_co = self.coattn['dcn_layers'][0](video_out, audio_out)
        
        # Joint fusion
        joint_input = torch.cat([video_out, audio_out, video_attended], dim=-1)
        joint_out = self.Joint(joint_input)  # (B, T, 512)
        
        # Predict valence and arousal
        valence = self.vregressor(joint_out)
        arousal = self.aregressor(joint_out)
        
        return valence, arousal


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def extract_video_features(video_path, device='cpu', target_fps=25, max_frames=300, per_frame=False, num_samples=None):
    """
    Extract visual features from video using ResNet50.
    
    Args:
        video_path: Path to video file
        device: 'cpu' or 'cuda'
        target_fps: Target frames per second (ignored if per_frame=True or num_samples is set)
        max_frames: Maximum number of frames to process
        per_frame: If True, extract features on every decoded frame (no downsampling)
        num_samples: If set, pick exactly this many frames evenly spread (for CPU sparse mode)
        
    Returns:
        features: numpy array of shape (num_frames, 512)
        actual_fps: actual FPS of the video
        frame_indices: numpy array of original frame indices (if per_frame=True)
    """
    print(f"  Extracting video features from: {video_path}")
    
    # Load ResNet50 for feature extraction
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove classifier
    resnet.eval()
    resnet.to(device)
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"    Video: {total_frames} frames, {fps:.1f} FPS, {duration:.2f}s duration")
    
    features_list = []
    frame_indices_list = []
    
    # Pre-calculate sampling if num_samples is set
    target_indices = None
    if num_samples is not None and not per_frame:
        if num_samples >= total_frames:
            target_indices = set(range(total_frames))
        else:
            target_indices = set(np.linspace(0, total_frames - 1, num_samples, dtype=int))
        print(f"    Sparse sampling: picking {len(target_indices)} specific frames across duration")

    frame_idx = 0
    if per_frame:
        # Extract features on every frame (no downsampling)
        print(f"    Extracting features on every frame (per_frame=True)")
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Preprocess and extract features
                input_tensor = preprocess(frame_rgb).unsqueeze(0).to(device)
                feature = resnet(input_tensor)
                feature = feature.squeeze().cpu().numpy()
                
                # ResNet outputs 2048-dim, we need to reduce to 512
                # Use a simple average pool over channel groups
                if feature.shape[0] == 2048:
                    feature = feature.reshape(4, 512).mean(axis=0)
                
                features_list.append(feature)
                frame_indices_list.append(frame_idx)
                frame_idx += 1
    else:
        # Original behavior: sample frames
        frame_skip = max(1, int(fps / target_fps))
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Decide if we skip this frame
                should_process = False
                if target_indices is not None:
                    should_process = (frame_idx in target_indices)
                else:
                    should_process = (frame_idx % frame_skip == 0)

                if should_process:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Preprocess and extract features
                    input_tensor = preprocess(frame_rgb).unsqueeze(0).to(device)
                    feature = resnet(input_tensor)
                    feature = feature.squeeze().cpu().numpy()
                    
                    # ResNet outputs 2048-dim, we need to reduce to 512
                    # Use a simple average pool over channel groups
                    if feature.shape[0] == 2048:
                        feature = feature.reshape(4, 512).mean(axis=0)
                    
                    features_list.append(feature)
                    frame_indices_list.append(frame_idx)
                    
                frame_idx += 1
                if target_indices is None and len(features_list) >= max_frames:
                    print(f"    Reached max_frames ({max_frames}) \u2014 stopping.")
                    break
    
    cap.release()
    
    features = np.array(features_list)
    frame_indices = np.array(frame_indices_list) if per_frame else None
    print(f"    Extracted {features.shape[0]} frame features, shape: {features.shape}")
    
    if per_frame:
        return features, fps, frame_indices
    return features, fps


def extract_audio_features(video_path, sr=16000, n_mels=128, hop_length=160, max_frames=300, 
                           video_fps=None, num_video_frames=None, per_frame=False):
    """
    Extract log-mel spectrogram features from video audio.
    
    Args:
        video_path: Path to video file
        sr: Sample rate
        n_mels: Number of mel bands
        hop_length: Hop length for STFT
        max_frames: Maximum number of frames (ignored if per_frame=True)
        video_fps: Video FPS (used for alignment if per_frame=True)
        num_video_frames: Number of video frames (used for alignment if per_frame=True)
        per_frame: If True, align audio features to match video frame count
        
    Returns:
        features: numpy array of shape (num_frames, 128) - log-mel features (no repeat trick)
    """
    print(f"  Extracting audio features from: {video_path}")
    
    try:
        # Load audio from video
        y, actual_sr = librosa.load(str(video_path), sr=sr, mono=True)
        
        if len(y) == 0:
            print("    Warning: No audio found, using zeros")
            if per_frame and num_video_frames is not None:
                return np.zeros((num_video_frames, n_mels))
            return np.zeros((max_frames, n_mels))
        
        duration = len(y) / sr
        print(f"    Audio: {duration:.2f}s duration, {sr} Hz sample rate")
        
        # Compute log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to (time, n_mels)
        log_mel = log_mel.T
        
        if per_frame and num_video_frames is not None and video_fps is not None:
            # Align audio features to match video frame count
            # Audio frame rate: sr / hop_length frames per second
            audio_fps = sr / hop_length
            target_frames = num_video_frames
            
            # Interpolate audio features to match video frame count
            if log_mel.shape[0] != target_frames:
                indices = np.linspace(0, log_mel.shape[0] - 1, target_frames)
                indices_floor = np.floor(indices).astype(int)
                indices_ceil = np.minimum(indices_floor + 1, log_mel.shape[0] - 1)
                weights = indices - indices_floor
                log_mel = (log_mel[indices_floor] * (1 - weights[:, None]) + 
                          log_mel[indices_ceil] * weights[:, None])
            
            features = log_mel
        else:
            # Limit to max_frames (original behavior)
            if log_mel.shape[0] > max_frames:
                features = log_mel[:max_frames]
            else:
                features = log_mel
        
        print(f"    Extracted {features.shape[0]} audio frames, shape: {features.shape}")
        
        return features
        
    except Exception as e:
        print(f"    Warning: Could not extract audio ({e}), using zeros")
        if per_frame and num_video_frames is not None:
            return np.zeros((num_video_frames, n_mels))
        return np.zeros((max_frames, n_mels))


def align_features(video_features, audio_features):
    """
    Align video and audio features to the same length.
    
    Args:
        video_features: (num_video_frames, 512)
        audio_features: (num_audio_frames, 1024)
        
    Returns:
        video_aligned: (seq_len, 512)
        audio_aligned: (seq_len, 1024)
    """
    seq_len = min(len(video_features), len(audio_features))
    
    if seq_len == 0:
        seq_len = max(len(video_features), len(audio_features), 1)
        if len(video_features) == 0:
            video_features = np.zeros((seq_len, 512))
        if len(audio_features) == 0:
            audio_features = np.zeros((seq_len, 1024))
    
    # Simple resampling using linear interpolation
    def resample(features, target_len):
        if len(features) == target_len:
            return features
        indices = np.linspace(0, len(features) - 1, target_len)
        indices_floor = np.floor(indices).astype(int)
        indices_ceil = np.minimum(indices_floor + 1, len(features) - 1)
        weights = indices - indices_floor
        return features[indices_floor] * (1 - weights[:, None]) + features[indices_ceil] * weights[:, None]
    
    video_aligned = resample(video_features, seq_len)
    audio_aligned = resample(audio_features, seq_len)
    
    return video_aligned, audio_aligned


# ============================================================================
# Model Loading and Inference
# ============================================================================

def load_model(model_path, device='cpu', use_audio_embedding=False):
    """
    Load the JointCAM model from checkpoint.
    
    Args:
        model_path: Path to jointcam_model.pt
        device: 'cpu' or 'cuda'
        use_audio_embedding: Whether to enable audio embedding (new preprocessing)
        
    Returns:
        model: Loaded JointCAM model
    """
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model with audio embedding if requested
    model = JointCAM(use_audio_embedding=use_audio_embedding)
    
    # Map checkpoint keys to model keys
    state_dict = checkpoint['net']
    model_state = model.state_dict()
    loaded_count = 0
    
    for key in model_state.keys():
        if key in state_dict and model_state[key].shape == state_dict[key].shape:
            model_state[key] = state_dict[key]
            loaded_count += 1
    
    model.load_state_dict(model_state, strict=False)
    print(f"  Loaded {loaded_count}/{len(model_state)} parameters")
    
    model.to(device)
    model.eval()
    
    return model


def predict_emotions(model, video_features, audio_features, device='cpu', 
                     window_size=300, stride=150, use_sliding_window=False):
    """
    Run inference to predict valence and arousal.
    
    Args:
        model: JointCAM model
        video_features: (seq_len, 512)
        audio_features: (seq_len, 1024)
        device: 'cpu' or 'cuda'
        window_size: Size of sliding window (default: 300)
        stride: Stride for sliding window (default: 150)
        use_sliding_window: If True, use sliding window for long sequences
        
    Returns:
        valence: numpy array of valence predictions
        arousal: numpy array of arousal predictions
    """
    seq_len = len(video_features)
    
    if not use_sliding_window or seq_len <= window_size:
        # Direct inference for short sequences
        video_tensor = torch.FloatTensor(video_features).unsqueeze(0).to(device)
        audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            valence, arousal = model(video_tensor, audio_tensor)
        
        valence = valence.squeeze().cpu().numpy()
        arousal = arousal.squeeze().cpu().numpy()
        
        return valence, arousal
    
    # Sliding window inference for long sequences
    print(f"    Using sliding window inference: window_size={window_size}, stride={stride}")
    
    # Initialize output arrays
    valence_all = np.zeros(seq_len)
    arousal_all = np.zeros(seq_len)
    count_all = np.zeros(seq_len)  # Track how many predictions per frame
    
    # Slide window over sequence
    start_idx = 0
    while start_idx < seq_len:
        end_idx = min(start_idx + window_size, seq_len)
        
        # Extract window
        video_window = video_features[start_idx:end_idx]
        audio_window = audio_features[start_idx:end_idx]
        
        # Pad if necessary
        if len(video_window) < window_size:
            pad_len = window_size - len(video_window)
            video_window = np.pad(video_window, ((0, pad_len), (0, 0)), mode='edge')
            audio_window = np.pad(audio_window, ((0, pad_len), (0, 0)), mode='edge')
        
        # Run inference on window
        video_tensor = torch.FloatTensor(video_window).unsqueeze(0).to(device)
        audio_tensor = torch.FloatTensor(audio_window).unsqueeze(0).to(device)
        
        with torch.no_grad():
            valence_win, arousal_win = model(video_tensor, audio_tensor)
        
        valence_win = valence_win.squeeze().cpu().numpy()
        arousal_win = arousal_win.squeeze().cpu().numpy()
        
        # Trim padding if applied
        actual_len = end_idx - start_idx
        valence_win = valence_win[:actual_len]
        arousal_win = arousal_win[:actual_len]
        
        # Accumulate predictions (averaging overlaps)
        valence_all[start_idx:end_idx] += valence_win
        arousal_all[start_idx:end_idx] += arousal_win
        count_all[start_idx:end_idx] += 1
        
        # Move window
        start_idx += stride
        
        # Break if we've covered everything
        if end_idx >= seq_len:
            break
    
    # Average overlapping predictions
    mask = count_all > 0
    valence_all[mask] /= count_all[mask]
    arousal_all[mask] /= count_all[mask]
    
    return valence_all, arousal_all


def interpret_emotions(valence, arousal):
    """
    Interpret valence/arousal values into emotion labels.
    
    Args:
        valence: Mean valence value [-1, 1]
        arousal: Mean arousal value [-1, 1]
        
    Returns:
        primary_emotion: Primary emotion label
        description: Description of the emotional state
    """
    # High Arousal, High Valence: Excited, Happy
    # High Arousal, Low Valence: Angry, Afraid
    # Low Arousal, High Valence: Calm, Relaxed
    # Low Arousal, Low Valence: Sad, Depressed
    
    if valence >= 0 and arousal >= 0:
        if valence > arousal:
            return "Happy", "Positive state with moderate arousal"
        else:
            return "Excited", "High positive arousal"
    elif valence < 0 and arousal >= 0:
        if abs(valence) > arousal:
            return "Angry", "Negative state with high arousal"
        else:
            return "Afraid", "Anxious or fearful high arousal"
    elif valence >= 0 and arousal < 0:
        if valence > abs(arousal):
            return "Calm", "Positive state with low arousal"
        else:
            return "Relaxed", "Deep relaxation or peacefulness"
    else:  # valence < 0 and arousal < 0
        if abs(valence) > abs(arousal):
            return "Sad", "Negative state with low arousal"
        else:
            return "Depressed", "Very low arousal negative state"


# ============================================================================
# Main Loop (Testing)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Video Emotion Detection using JointCAM')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--model_path', type=str, default='models/jointcam_finetuned_v4.pt',
                        help='Path to pre-trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
        
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return

    # Load model
    device = args.device
    print(f"Using device: {device}")
    model = load_model(args.model_path, device)
    
    # Feature extraction
    video_features, fps = extract_video_features(args.video_path, device)
    audio_features = extract_audio_features(args.video_path)
    
    # Alignment
    video_aligned, audio_aligned = align_features(video_features, audio_features)
    
    # Inference
    valence, arousal = predict_emotions(model, video_aligned, audio_aligned, device)
    
    # Result interpretation (overall)
    mean_v = np.mean(valence)
    mean_a = np.mean(arousal)
    emotion, desc = interpret_emotions(mean_v, mean_a)
    
    print("\n" + "="*50)
    print(f"ANALYSIS COMPLETE")
    print("="*50)
    print(f"  Overall Valence: {mean_v:+.4f}")
    print(f"  Overall Arousal: {mean_a:+.4f}")
    print(f"  Detected Emotion: {emotion}")
    print(f"  Description: {desc}")
    print("="*50)

if __name__ == '__main__':
    main()
