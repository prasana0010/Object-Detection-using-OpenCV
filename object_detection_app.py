import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from collections import deque, defaultdict
import json
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

# Import YOLO (you'll need to install ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics not found. Please install with: pip install ultralytics")

class AdvancedObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        
        # Initialize variables
        self.model = None
        self.is_detecting = False
        self.current_frame = None
        self.detection_thread = None
        self.performance_data = deque(maxlen=100)
        self.class_counts = defaultdict(int)
        self.total_detections = 0
        self.session_start_time = time.time()
        
        # Create the interface
        self.create_modern_interface()
        self.load_default_model()
        
    def setup_window(self):
        """Configure the main window with modern styling"""
        self.root.title("🎯 Advanced YOLO Object Detection Suite")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#0d1117')  # GitHub dark theme
        
        # Configure ttk styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Custom color scheme
        self.colors = {
            'bg_primary': '#0d1117',
            'bg_secondary': '#21262d', 
            'bg_tertiary': '#30363d',
            'accent_blue': '#58a6ff',
            'accent_green': '#3fb950',
            'accent_orange': '#f85149',
            'accent_purple': '#bc8cff',
            'text_primary': '#f0f6fc',
            'text_secondary': '#8b949e'
        }
        
        # Configure styles
        self.style.configure('Modern.TFrame', background=self.colors['bg_secondary'])
        self.style.configure('Card.TFrame', background=self.colors['bg_tertiary'])
        self.style.configure('Title.TLabel', 
                           font=('Segoe UI', 24, 'bold'),
                           background=self.colors['bg_primary'],
                           foreground=self.colors['accent_blue'])
        self.style.configure('Header.TLabel',
                           font=('Segoe UI', 14, 'bold'),
                           background=self.colors['bg_secondary'],
                           foreground=self.colors['text_primary'])
        self.style.configure('Info.TLabel',
                           font=('Segoe UI', 10),
                           background=self.colors['bg_tertiary'],
                           foreground=self.colors['text_secondary'])
        self.style.configure('Success.TLabel',
                           foreground=self.colors['accent_green'])
        self.style.configure('Warning.TLabel',
                           foreground=self.colors['accent_orange'])
        
    def create_modern_interface(self):
        """Create the modern, vibrant user interface"""
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title section
        self.create_title_section(main_container)
        
        # Main content area
        content_frame = tk.Frame(main_container, bg=self.colors['bg_primary'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Left panel - Controls
        self.create_control_panel(content_frame)
        
        # Center panel - Video display
        self.create_display_panel(content_frame)
        
        # Right panel - Statistics
        self.create_stats_panel(content_frame)
        
        # Bottom panel - Status and progress
        self.create_status_panel(main_container)
        
    def create_title_section(self, parent):
        """Create animated title section"""
        title_frame = tk.Frame(parent, bg=self.colors['bg_primary'])
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Animated title with gradient effect simulation
        title_label = tk.Label(title_frame,
                              text="🎯 ADVANCED YOLO OBJECT DETECTION SUITE",
                              font=('Segoe UI', 28, 'bold'),
                              bg=self.colors['bg_primary'],
                              fg=self.colors['accent_blue'])
        title_label.pack()
        
        # Subtitle
        subtitle = tk.Label(title_frame,
                           text="Real-time AI-powered object detection with advanced analytics",
                           font=('Segoe UI', 12),
                           bg=self.colors['bg_primary'],
                           fg=self.colors['text_secondary'])
        subtitle.pack(pady=(5, 0))
        
        # Separator line
        separator = tk.Frame(title_frame, height=3, bg=self.colors['accent_blue'])
        separator.pack(fill=tk.X, pady=(10, 0))
        
    def create_control_panel(self, parent):
        """Create the left control panel"""
        control_frame = tk.Frame(parent, bg=self.colors['bg_secondary'], relief=tk.RAISED, bd=1)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Panel title
        tk.Label(control_frame, text="🎛️ CONTROL PANEL",
                font=('Segoe UI', 16, 'bold'),
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary']).pack(pady=15)
        
        # Model selection section
        model_section = tk.LabelFrame(control_frame, text="🤖 Model Configuration",
                                     bg=self.colors['bg_secondary'],
                                     fg=self.colors['text_primary'],
                                     font=('Segoe UI', 11, 'bold'))
        model_section.pack(fill=tk.X, padx=15, pady=10)
        
        # Model dropdown
        tk.Label(model_section, text="YOLO Model:",
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_secondary']).pack(anchor='w', padx=10, pady=(10, 5))
        
        self.model_var = tk.StringVar(value="yolov8n.pt")
        model_combo = ttk.Combobox(model_section, textvariable=self.model_var,
                                  values=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
                                         'yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10l.pt',
                                         'yolov11n.pt', 'yolov11s.pt', 'yolov11m.pt'])
        model_combo.pack(fill=tk.X, padx=10, pady=(0, 10))
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Confidence threshold
        tk.Label(model_section, text="Confidence Threshold:",
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_secondary']).pack(anchor='w', padx=10, pady=(5, 5))
        
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = tk.Scale(model_section, from_=0.1, to=1.0, 
                                   resolution=0.05, orient=tk.HORIZONTAL,
                                   variable=self.confidence_var,
                                   bg=self.colors['bg_secondary'],
                                   fg=self.colors['text_primary'],
                                   highlightbackground=self.colors['bg_secondary'])
        confidence_scale.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Detection modes section
        mode_section = tk.LabelFrame(control_frame, text="📹 Detection Mode",
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text_primary'],
                                    font=('Segoe UI', 11, 'bold'))
        mode_section.pack(fill=tk.X, padx=15, pady=10)
        
        # Mode buttons with modern styling
        self.create_modern_button(mode_section, "📷 Webcam", self.start_webcam_detection, 
                                 self.colors['accent_green'])
        self.create_modern_button(mode_section, "🖼️ Image", self.detect_image, 
                                 self.colors['accent_blue'])
        self.create_modern_button(mode_section, "🎬 Video", self.detect_video, 
                                 self.colors['accent_purple'])
        self.create_modern_button(mode_section, "📁 Batch", self.batch_detection, 
                                 self.colors['accent_orange'])
        
        # Control buttons
        control_section = tk.LabelFrame(control_frame, text="⚡ Controls",
                                       bg=self.colors['bg_secondary'],
                                       fg=self.colors['text_primary'],
                                       font=('Segoe UI', 11, 'bold'))
        control_section.pack(fill=tk.X, padx=15, pady=10)
        
        self.stop_btn = self.create_modern_button(control_section, "⏹️ Stop", self.stop_detection, 
                                                 self.colors['accent_orange'])
        self.stop_btn.config(state='disabled')
        
        self.create_modern_button(control_section, "📊 Export Report", self.export_report, 
                                 self.colors['accent_blue'])
        self.create_modern_button(control_section, "🗑️ Clear Stats", self.clear_stats, 
                                 self.colors['text_secondary'])
        
    def create_modern_button(self, parent, text, command, color):
        """Create a modern styled button"""
        btn = tk.Button(parent, text=text, command=command,
                       bg=color, fg='white',
                       font=('Segoe UI', 10, 'bold'),
                       relief=tk.FLAT, bd=0, pady=8,
                       cursor='hand2')
        btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Add hover effects
        def on_enter(e):
            btn.configure(bg=self.lighten_color(color))
        def on_leave(e):
            btn.configure(bg=color)
            
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        
        return btn
        
    def lighten_color(self, color):
        """Lighten a hex color for hover effects"""
        # Simple color lightening - in a full implementation you'd use proper color manipulation
        color_map = {
            self.colors['accent_green']: '#4fbf5f',
            self.colors['accent_blue']: '#68b6ff',
            self.colors['accent_purple']: '#cc9cff',
            self.colors['accent_orange']: '#ff6159',
            self.colors['text_secondary']: '#9ba5ae'
        }
        return color_map.get(color, color)
        
    def create_display_panel(self, parent):
        """Create the center display panel"""
        display_frame = tk.Frame(parent, bg=self.colors['bg_secondary'], relief=tk.RAISED, bd=1)
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Display title
        tk.Label(display_frame, text="🎬 LIVE DETECTION DISPLAY",
                font=('Segoe UI', 16, 'bold'),
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary']).pack(pady=15)
        
        # Video display area
        self.video_frame = tk.Frame(display_frame, bg='black', relief=tk.SUNKEN, bd=2)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Video label
        self.video_label = tk.Label(self.video_frame, 
                                   text="🎥 Select detection mode to begin...",
                                   bg='black', fg='white',
                                   font=('Segoe UI', 14))
        self.video_label.pack(expand=True)
        
        # Performance overlay
        self.create_performance_overlay(display_frame)
        
    def create_performance_overlay(self, parent):
        """Create performance monitoring overlay"""
        perf_frame = tk.Frame(parent, bg=self.colors['bg_tertiary'])
        perf_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Performance metrics
        metrics_frame = tk.Frame(perf_frame, bg=self.colors['bg_tertiary'])
        metrics_frame.pack(fill=tk.X, pady=10)
        
        # FPS display
        self.fps_label = tk.Label(metrics_frame, text="FPS: 0.0",
                                 bg=self.colors['bg_tertiary'],
                                 fg=self.colors['accent_green'],
                                 font=('Segoe UI', 12, 'bold'))
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        # Detection count
        self.detection_label = tk.Label(metrics_frame, text="Detections: 0",
                                       bg=self.colors['bg_tertiary'],
                                       fg=self.colors['accent_blue'],
                                       font=('Segoe UI', 12, 'bold'))
        self.detection_label.pack(side=tk.LEFT, padx=10)
        
        # Processing time
        self.processing_label = tk.Label(metrics_frame, text="Processing: 0ms",
                                        bg=self.colors['bg_tertiary'],
                                        fg=self.colors['accent_purple'],
                                        font=('Segoe UI', 12, 'bold'))
        self.processing_label.pack(side=tk.LEFT, padx=10)
        
    def create_stats_panel(self, parent):
        """Create the right statistics panel"""
        stats_frame = tk.Frame(parent, bg=self.colors['bg_secondary'], relief=tk.RAISED, bd=1)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Stats title
        tk.Label(stats_frame, text="📊 ANALYTICS DASHBOARD",
                font=('Segoe UI', 16, 'bold'),
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary']).pack(pady=15)
        
        # Real-time chart
        self.create_performance_chart(stats_frame)
        
        # Class detection counts
        self.create_class_stats(stats_frame)
        
        # Session information
        self.create_session_info(stats_frame)
        
    def create_performance_chart(self, parent):
        """Create real-time performance chart"""
        chart_frame = tk.LabelFrame(parent, text="⚡ Performance Metrics",
                                   bg=self.colors['bg_secondary'],
                                   fg=self.colors['text_primary'],
                                   font=('Segoe UI', 11, 'bold'))
        chart_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Create matplotlib figure
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(4, 3), facecolor=self.colors['bg_secondary'])
        self.ax.set_facecolor(self.colors['bg_tertiary'])
        
        # Initial empty plot
        self.fps_line, = self.ax.plot([], [], color=self.colors['accent_green'], linewidth=2, label='FPS')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 60)
        self.ax.set_xlabel('Time', color='white')
        self.ax.set_ylabel('FPS', color='white')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # Embed in tkinter
        self.chart_canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_class_stats(self, parent):
        """Create class detection statistics"""
        class_frame = tk.LabelFrame(parent, text="🎯 Detection Statistics",
                                   bg=self.colors['bg_secondary'],
                                   fg=self.colors['text_primary'],
                                   font=('Segoe UI', 11, 'bold'))
        class_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Scrollable text for class counts
        self.class_text = tk.Text(class_frame, height=8, width=25,
                                 bg=self.colors['bg_tertiary'],
                                 fg=self.colors['text_primary'],
                                 font=('Consolas', 9))
        scrollbar = tk.Scrollbar(class_frame, orient=tk.VERTICAL, command=self.class_text.yview)
        self.class_text.configure(yscrollcommand=scrollbar.set)
        
        self.class_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_session_info(self, parent):
        """Create session information panel"""
        session_frame = tk.LabelFrame(parent, text="ℹ️ Session Info",
                                     bg=self.colors['bg_secondary'],
                                     fg=self.colors['text_primary'],
                                     font=('Segoe UI', 11, 'bold'))
        session_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Session labels
        self.session_time_label = tk.Label(session_frame, text="Duration: 00:00:00",
                                          bg=self.colors['bg_secondary'],
                                          fg=self.colors['text_secondary'],
                                          font=('Segoe UI', 10))
        self.session_time_label.pack(anchor='w', padx=10, pady=2)
        
        self.total_detections_label = tk.Label(session_frame, text="Total Detections: 0",
                                              bg=self.colors['bg_secondary'],
                                              fg=self.colors['text_secondary'],
                                              font=('Segoe UI', 10))
        self.total_detections_label.pack(anchor='w', padx=10, pady=2)
        
        self.avg_fps_label = tk.Label(session_frame, text="Avg FPS: 0.0",
                                     bg=self.colors['bg_secondary'],
                                     fg=self.colors['text_secondary'],
                                     font=('Segoe UI', 10))
        self.avg_fps_label.pack(anchor='w', padx=10, pady=2)
        
    def create_status_panel(self, parent):
        """Create bottom status panel"""
        status_frame = tk.Frame(parent, bg=self.colors['bg_secondary'], relief=tk.RAISED, bd=1)
        status_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Status label
        self.status_label = tk.Label(status_frame, text="🟢 Ready - System initialized successfully",
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['accent_green'],
                                    font=('Segoe UI', 11, 'bold'))
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=20, pady=10, fill=tk.X, expand=False)
        
    def load_default_model(self):
        """Load the default YOLO model"""
        if not YOLO_AVAILABLE:
            self.status_label.config(text="❌ YOLO not available - Please install ultralytics", 
                                   fg=self.colors['accent_orange'])
            return
            
        try:
            self.status_label.config(text="🔄 Loading YOLO model...", 
                                   fg=self.colors['accent_blue'])
            self.model = YOLO('yolov8n.pt')
            self.status_label.config(text="✅ YOLO model loaded successfully", 
                                   fg=self.colors['accent_green'])
        except Exception as e:
            self.status_label.config(text=f"❌ Error loading model: {str(e)}", 
                                   fg=self.colors['accent_orange'])
            
    def on_model_change(self, event=None):
        """Handle model change"""
        if not YOLO_AVAILABLE:
            return
            
        try:
            model_name = self.model_var.get()
            self.status_label.config(text=f"🔄 Loading {model_name}...", 
                                   fg=self.colors['accent_blue'])
            self.model = YOLO(model_name)
            self.status_label.config(text=f"✅ {model_name} loaded successfully", 
                                   fg=self.colors['accent_green'])
        except Exception as e:
            self.status_label.config(text=f"❌ Error loading {model_name}: {str(e)}", 
                                   fg=self.colors['accent_orange'])
            
    def start_webcam_detection(self):
        """Start real-time webcam detection"""
        if not self.model:
            messagebox.showerror("Error", "Please load a YOLO model first!")
            return
            
        if self.is_detecting:
            return
            
        self.is_detecting = True
        self.stop_btn.config(state='normal')
        self.progress.start(10)
        
        self.detection_thread = threading.Thread(target=self.webcam_detection_worker, daemon=True)
        self.detection_thread.start()
        
    def webcam_detection_worker(self):
        """Worker thread for webcam detection"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps_counter = deque(maxlen=30)
        
        while self.is_detecting:
            ret, frame = cap.read()
            if not ret:
                continue
                
            start_time = time.time()
            
            # Run YOLO detection
            results = self.model(frame, conf=self.confidence_var.get(), verbose=False)
            
            # Process results
            detection_count = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    detection_count = len(boxes)
                    for box in boxes:
                        # Get class name and update counts
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        self.class_counts[class_name] += 1
                        self.total_detections += 1
                        
            # Draw results
            annotated_frame = results[0].plot() if results else frame
            
            # Calculate FPS
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            fps_counter.append(fps)
            
            # Update performance data
            self.performance_data.append(fps)
            
            # Update UI
            self.root.after(0, self.update_display, annotated_frame, fps, detection_count, processing_time * 1000)
            
        cap.release()
        
    def update_display(self, frame, fps, detections, processing_time):
        """Update the display with new frame and stats"""
        # Convert frame to PhotoImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Resize to fit display
        display_width = self.video_frame.winfo_width() - 20
        display_height = self.video_frame.winfo_height() - 20
        
        if display_width > 100 and display_height > 100:
            image = image.resize((min(display_width, 600), min(display_height, 450)), Image.Resampling.LANCZOS)
            
        photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo
        
        # Update performance metrics
        self.fps_label.config(text=f"FPS: {fps:.1f}")
        self.detection_label.config(text=f"Detections: {detections}")
        self.processing_label.config(text=f"Processing: {processing_time:.1f}ms")
        
        # Update chart
        self.update_performance_chart()
        
        # Update class statistics
        self.update_class_stats()
        
        # Update session info
        self.update_session_info()
        
    def update_performance_chart(self):
        """Update the real-time performance chart"""
        if len(self.performance_data) > 1:
            x_data = list(range(len(self.performance_data)))
            y_data = list(self.performance_data)
            
            self.fps_line.set_data(x_data, y_data)
            
            if len(self.performance_data) > 100:
                self.ax.set_xlim(len(self.performance_data) - 100, len(self.performance_data))
            
            max_fps = max(y_data) if y_data else 60
            self.ax.set_ylim(0, max(max_fps + 5, 30))
            
            self.chart_canvas.draw_idle()
            
    def update_class_stats(self):
        """Update class detection statistics"""
        self.class_text.delete(1.0, tk.END)
        
        if self.class_counts:
            stats_text = "Detection Counts:\n" + "="*20 + "\n"
            sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
            
            for class_name, count in sorted_classes[:10]:  # Show top 10
                stats_text += f"{class_name:<12}: {count:>4}\n"
                
            self.class_text.insert(1.0, stats_text)
        else:
            self.class_text.insert(1.0, "No detections yet...")
            
    def update_session_info(self):
        """Update session information"""
        # Calculate session duration
        duration = time.time() - self.session_start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        self.session_time_label.config(text=f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.total_detections_label.config(text=f"Total Detections: {self.total_detections}")
        
        avg_fps = np.mean(self.performance_data) if self.performance_data else 0
        self.avg_fps_label.config(text=f"Avg FPS: {avg_fps:.1f}")
        
    def detect_image(self):
        """Detect objects in a single image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path and self.model:
            try:
                # Load and process image
                image = cv2.imread(file_path)
                results = self.model(image, conf=self.confidence_var.get())
                
                # Display result
                annotated_image = results[0].plot()
                self.display_static_result(annotated_image, "Image Detection Result")
                
                # Update statistics
                detection_count = len(results[0].boxes) if results[0].boxes else 0
                self.status_label.config(text=f"✅ Image processed - {detection_count} objects detected")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image: {str(e)}")
                
    def detect_video(self):
        """Detect objects in a video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path and self.model:
            # This would implement video processing
            messagebox.showinfo("Info", "Video detection feature - Implementation depends on requirements")
            
    def batch_detection(self):
        """Batch process multiple images"""
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        
        if folder_path and self.model:
            # This would implement batch processing
            messagebox.showinfo("Info", "Batch detection feature - Implementation depends on requirements")
            
    def display_static_result(self, image, title):
        """Display a static result in a new window"""
        result_window = tk.Toplevel(self.root)
        result_window.title(title)
        result_window.configure(bg=self.colors['bg_primary'])
        
        # Convert and display image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_image = pil_image.resize((800, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_image)
        
        label = tk.Label(result_window, image=photo, bg=self.colors['bg_primary'])
        label.image = photo
        label.pack(padx=20, pady=20)
        
    def stop_detection(self):
        """Stop current detection process"""
        self.is_detecting = False
        self.stop_btn.config(state='disabled')
        self.progress.stop()
        self.status_label.config(text="⏹️ Detection stopped", fg=self.colors['accent_orange'])
        
    def clear_stats(self):
        """Clear all statistics"""
        self.class_counts.clear()
        self.performance_data.clear()
        self.total_detections = 0
        self.session_start_time = time.time()
        
        # Clear displays
        self.class_text.delete(1.0, tk.END)
        self.class_text.insert(1.0, "Statistics cleared...")
        
        self.status_label.config(text="🗑️ Statistics cleared", fg=self.colors['accent_blue'])
        
    def export_report(self):
        """Export detection report"""
        file_path = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_var.get(),
                    "confidence_threshold": self.confidence_var.get(),
                    "total_detections": self.total_detections,
                    "session_duration": time.time() - self.session_start_time,
                    "class_counts": dict(self.class_counts),
                    "average_fps": float(np.mean(self.performance_data)) if self.performance_data else 0
                }
                
                with open(file_path, 'w') as f:
                    json.dump(report, f, indent=2)
                    
                self.status_label.config(text=f"📊 Report exported to {os.path.basename(file_path)}", 
                                       fg=self.colors['accent_green'])
                messagebox.showinfo("Success", f"Report exported successfully to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting report: {str(e)}")

def main():
    root = tk.Tk()
    app = AdvancedObjectDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
