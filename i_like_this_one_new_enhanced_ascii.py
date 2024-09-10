import cv2
import numpy as np
import pygame
import sys
import time
import pyaudio
import struct
import colorsys

ASCII_CHARS = ['@', '#', 'S', '%', '?', '*', '+', ';', ':', ',', '.', ' ']

def testDevice(source):
    cap = cv2.VideoCapture(source) 
    if cap is None or not cap.isOpened():
        return False
    return True

def resize_image(image, new_width=120):
    height, width = image.shape[:2]
    ratio = height / width
    new_height = int(new_width * ratio)
    return cv2.resize(image, (new_width, new_height))

def apply_dynamic_contrast(image, contrast_factor):
    mean = np.mean(image)
    return np.clip((image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

def enhance_edges(image):
    edges = cv2.Canny(image, 100, 200)
    return cv2.addWeighted(image, 0.7, edges, 0.3, 0)

def adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def amplify_brightness(image, amplification_factor):
    return np.clip(image * amplification_factor, 0, 255).astype(np.uint8)

def pixels_to_ascii(image):
    return np.array([[ASCII_CHARS[int((pixel/255.0) * (len(ASCII_CHARS)-1))] for pixel in row] for row in image])

class AudioInput:
    def __init__(self):
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)

    def get_audio_data(self):
        try:
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            return struct.unpack(str(self.CHUNK) + 'h', data)
        except OSError as e:
            print(f"Audio Error: {e}")
            return [0] * self.CHUNK

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

def get_frequency_bands(audio_data, rate, num_bands=6):
    fft_data = np.fft.fft(audio_data)
    freq_magnitudes = np.abs(fft_data[:len(fft_data)//2])
    freq_bins = np.fft.fftfreq(len(audio_data), 1/rate)[:len(freq_magnitudes)]
    
    max_freq = rate // 2
    band_width = max_freq / num_bands
    bands = []
    
    for i in range(num_bands):
        start_freq = i * band_width
        end_freq = (i + 1) * band_width
        band_magnitudes = freq_magnitudes[(freq_bins >= start_freq) & (freq_bins < end_freq)]
        bands.append(np.mean(band_magnitudes))
    
    return [min(1.0, band / np.max(bands)) for band in bands]

def get_dynamic_color(bands):
    hue = (bands[0] * 360) % 360
    saturation = 0.5 + bands[1] * 0.5
    value = 0.5 + bands[2] * 0.5
    r, g, b = colorsys.hsv_to_rgb(hue / 360, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))

def apply_color_depth(surface, color, depth):
    colored_surface = surface.copy()
    colored_surface.fill(color, special_flags=pygame.BLEND_RGBA_MULT)
    result = surface.copy()
    for _ in range(depth):
        result.blit(colored_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    return result

def main():
    pygame.init()

    camDevice = None
    for cam in range(0,2):
        if testDevice(cam):
            camDevice = cam
    if camDevice is None:
        print("Error: Could not find video capture device.")
        return

    width, height = 1200, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Static ASCII Video with Audio-Reactive Effects")

    ascii_font = pygame.font.SysFont('courier', 7)
    info_font = pygame.font.SysFont('arial', 20)

    cap = cv2.VideoCapture(camDevice)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return

    audio = AudioInput()

    clock = pygame.time.Clock()

    # Pre-render ASCII characters
    ascii_surfaces = {char: ascii_font.render(char, True, (255, 255, 255)) for char in ASCII_CHARS}

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                cap.release()
                audio.close()
                pygame.quit()
                sys.exit()

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video capture device.")
            break

        audio_data = audio.get_audio_data()
        frequency_bands = get_frequency_bands(audio_data, audio.RATE)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = resize_image(gray, new_width=120)

        # Apply dynamic contrast
        contrast_factor = 1 + frequency_bands[0] * 2  # Use first band for contrast
        contrasted = apply_dynamic_contrast(resized, contrast_factor)

        # Enhance edges
        edge_enhanced = enhance_edges(contrasted)

        # Apply adaptive thresholding
        thresholded = adaptive_threshold(edge_enhanced)

        # Amplify brightness
        amplification_factor = 1 + frequency_bands[1]  # Use second band for brightness
        brightened = amplify_brightness(thresholded, amplification_factor)

        ascii_image = pixels_to_ascii(brightened)

        ascii_height, ascii_width = ascii_image.shape

        screen.fill((0, 0, 0))

        color = get_dynamic_color(frequency_bands)
        
        for row in range(ascii_height):
            for col in range(ascii_width):
                char = ascii_image[row, col]
                x = col * 10
                y = row * 10
                
                size_factor = 1 + frequency_bands[3] * 0.5
                scaled_surface = pygame.transform.scale(ascii_surfaces[char], 
                                                        (int(10 * size_factor), int(10 * size_factor)))
                
                color_depth = int(frequency_bands[4] * 5) + 1
                colored_surface = apply_color_depth(scaled_surface, color, color_depth)
                
                # Add a glow effect
                glow_intensity = int(frequency_bands[5] * 155) + 100
                glow_surf = pygame.Surface((int(12 * size_factor), int(12 * size_factor)), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*color, glow_intensity), 
                                   (int(6 * size_factor), int(6 * size_factor)), 
                                   int(6 * size_factor))
                screen.blit(glow_surf, (x - int(size_factor), y - int(size_factor)))
                screen.blit(colored_surface, (x, y))

        for i, band in enumerate(frequency_bands):
            band_text = info_font.render(f"Band {i+1}: {band:.2f}", True, (255, 255, 255))
            screen.blit(band_text, (10, 10 + i * 30))

        pygame.display.flip()
        clock.tick(30)

if __name__ == '__main__':
    main()
