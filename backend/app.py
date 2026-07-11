"""
Pushkarani Classification & Water Quality Analysis Backend
Comprehensive Flask API with Model Serving, Water Quality Analysis, and Chatbot
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_caching import Cache
import numpy as np
import json
import os
import io
from PIL import Image
import base64
from datetime import datetime
import logging
from dotenv import load_dotenv
import pickle
import sys

# Import comprehensive chatbot knowledge base
try:
    from chatbot_kb import KNOWLEDGE_BASE
except ImportError:
    KNOWLEDGE_BASE = {}

# Defer TensorFlow import until needed
TF_AVAILABLE = False
keras = None

def ensure_tf_loaded():
    global TF_AVAILABLE, keras
    if not TF_AVAILABLE:
        try:
            import tensorflow as tf
            from tensorflow import keras as keras_module
            keras = keras_module
            TF_AVAILABLE = True
            logger.info("[OK] TensorFlow loaded on demand")
        except Exception as e:
            logger.error(f"TensorFlow load error: {e}")
            TF_AVAILABLE = False

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Cache configuration
cache_config = {'CACHE_TYPE': 'simple'}
cache = Cache(app, config=cache_config)

# Add request logging
@app.before_request
def log_request():
    try:
        print(f"[REQUEST] {request.method} {request.path}", flush=True)
    except Exception as e:
        print(f"[ERROR in before_request] {e}", flush=True)

# Logging setup
logging.basicConfig(level=logging.ERROR, format='%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Configuration - OPTIMIZED FOR BEST MODELS ONLY
# Calculate base directory (parent of backend folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'IMG_HEIGHT': 224,
    'IMG_WIDTH': 224,
    'MODEL_PATHS': {
        # Available Models - Updated to match existing files
        'densenet': os.path.join(BASE_DIR, 'densenet', 'best_model.keras'),
        'mobilenet': os.path.join(BASE_DIR, 'mobilenet', 'final_model.keras'),
        'inception': os.path.join(BASE_DIR, 'InceptionV3', 'best_model_phase1.keras'),
        'mobilenetv3': os.path.join(BASE_DIR, 'MobileNetV3Small', 'best_model.keras'),
    },
    'STONE_QUALITY_MODEL_PATH': os.path.join(BASE_DIR, 'stone_quality_model', 'final_model.keras'),
    'CLASS_INDICES_PATH': os.path.join(BASE_DIR, 'densenet', 'class_indices.json'),
    'STONE_QUALITY_CLASSES': ['Bad', 'Good', 'Medium'],
    'BEST_MODELS': ['densenet', 'mobilenet', 'inception', 'mobilenetv3']
}

# Global models cache
loaded_models = {}
class_indices = None

# ==================== INITIALIZATION ====================

def load_all_models():
    """Load class indices and ensure TensorFlow is loaded"""
    global loaded_models, class_indices, TF_AVAILABLE, keras
    
    logger.info("Initializing system...")
    
    # Ensure TensorFlow is loaded at startup
    ensure_tf_loaded()
    
    # Load class indices only
    if os.path.exists(CONFIG['CLASS_INDICES_PATH']):
        with open(CONFIG['CLASS_INDICES_PATH'], 'r') as f:
            class_indices = json.load(f)
        logger.info(f"[OK] Classes loaded: {list(class_indices.keys())}")
    else:
        logger.warning(f"Class indices not found at {CONFIG['CLASS_INDICES_PATH']}")
        class_indices = {'type-1': 0, 'type-2': 1, 'type-3': 2}
    
    logger.info("[OK] System initialized (TensorFlow loaded, models on-demand)")
    return True

def load_single_model(model_name):
    """Load a single model on demand"""
    global loaded_models, TF_AVAILABLE, keras
    
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    # Ensure TensorFlow is loaded
    ensure_tf_loaded()
    
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available")
        return None
    
    if model_name not in CONFIG['MODEL_PATHS']:
        return None
    
    model_path = CONFIG['MODEL_PATHS'][model_name]
    if os.path.exists(model_path):
        try:
            model = keras.models.load_model(model_path)
            loaded_models[model_name] = model
            logger.info(f"[OK] {model_name.upper()} loaded successfully")
            return model
        except Exception as e:
            logger.error(f"[ERROR] Failed to load {model_name}: {str(e)}")
    return None

def load_stone_quality_model():
    """Load stone quality prediction model"""
    global loaded_models, TF_AVAILABLE, keras
    
    if 'stone_quality_model' in loaded_models:
        return loaded_models['stone_quality_model']
    
    ensure_tf_loaded()
    
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available")
        return None
    
    model_path = CONFIG['STONE_QUALITY_MODEL_PATH']
    if os.path.exists(model_path):
        try:
            model = keras.models.load_model(model_path)
            loaded_models['stone_quality_model'] = model
            logger.info(f"[OK] Stone quality model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"[ERROR] Failed to load stone quality model: {str(e)}")
    else:
        logger.warning(f"Stone quality model not found at {model_path}")
    return None

def predict_stone_quality(img_array):
    """Predict stone quality (Good, Bad, Medium) based on structural integrity"""
    try:
        stone_model = load_stone_quality_model()
        if stone_model is None:
            return None
        
        predictions = stone_model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx] * 100)
        
        # Map to stone quality classes
        stone_classes = CONFIG['STONE_QUALITY_CLASSES']
        predicted_quality = stone_classes[predicted_idx]
        
        logger.info(f"Stone quality prediction: {predicted_quality} ({confidence:.2f}%)")
        
        return {
            'quality': predicted_quality,
            'confidence': confidence,
            'all_predictions': {
                cls: float(predictions[0][i] * 100)
                for i, cls in enumerate(stone_classes)
            },
            'maintenance_required': predicted_quality in ['Bad', 'Medium'],
            'maintenance_urgency': 'high' if predicted_quality == 'Bad' else 'medium' if predicted_quality == 'Medium' else 'low'
        }
    except Exception as e:
        logger.error(f"[ERROR] Stone quality prediction error: {str(e)}")
        return None

# ==================== PREPROCESSING ====================

def preprocess_image(image_data):
    """Preprocess image for model prediction"""
    try:
        # Handle base64 encoded images
        if isinstance(image_data, str):
            # Remove data URI prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            image = Image.open(image_data)
        
        # Convert to RGB
        image = image.convert('RGB')
        original_img = image.copy()
        
        # Resize
        image = image.resize((CONFIG['IMG_WIDTH'], CONFIG['IMG_HEIGHT']))
        
        # Convert to array and normalize
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, original_img
    
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise

# ==================== STABILITY DETECTION ====================

# ==================== PREDICTION ====================

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict Pushkarani type using BEST models only
    Accepts: image file or base64 encoded image
    Query param: model (default: densenet) - must be densenet, inception, or mobilenetv2
    """
    try:
        print("[DEBUG] /api/predict route called", flush=True)
        
        # Get model selection (only available models allowed)
        model_name = request.args.get('model', 'densenet').lower()
        available_models = ['densenet', 'mobilenet', 'inception', 'mobilenetv3']
        
        if model_name not in available_models:
            return jsonify({
                'error': f'Model "{model_name}" not available. Use only: {available_models}',
                'available_models': available_models,
                'accuracies': {
                    'densenet': '96%',
                    'mobilenet': '96%',
                    'inception': '95%',
                    'mobilenetv3': '94%'
                }
            }), 400
        
        print(f"[DEBUG] Loading model: {model_name}", flush=True)
        
        # Load model on demand if not already loaded
        model = load_single_model(model_name)
        if model is None:
            print(f"[ERROR] Failed to load model: {model_name}", flush=True)
            return jsonify({
                'error': f'Failed to load model "{model_name}"',
                'available_models': best_models
            }), 500
        
        # Get image - check files FIRST to avoid 415 error
        print(f"[DEBUG] request.files keys: {list(request.files.keys())}", flush=True)
        print(f"[DEBUG] request.form keys: {list(request.form.keys())}", flush=True)
        
        if 'image' in request.files:
            print(f"[DEBUG] Found 'image' in request.files", flush=True)
            image_file = request.files['image']
            print(f"[DEBUG] image_file: {image_file}, type: {type(image_file)}", flush=True)
            img_array, original_img = preprocess_image(image_file)
        elif 'file' in request.files:
            print(f"[DEBUG] Found 'file' in request.files", flush=True)
            image_file = request.files['file']
            print(f"[DEBUG] image_file: {image_file}, type: {type(image_file)}", flush=True)
            img_array, original_img = preprocess_image(image_file)
        else:
            # Only try JSON if files are not present
            try:
                if request.json and 'image' in request.json:
                    img_array, original_img = preprocess_image(request.json['image'])
                else:
                    return jsonify({'error': 'No image provided'}), 400
            except:
                return jsonify({'error': 'No image provided'}), 400
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        
        # Create index to class mapping
        index_to_class = {v: k for k, v in class_indices.items()}
        predicted_class = index_to_class[predicted_idx]
        confidence = float(predictions[0][predicted_idx] * 100)
        
        # Get all probabilities
        all_probs = {
            index_to_class[i]: float(predictions[0][i] * 100)
            for i in range(len(predictions[0]))
        }
        
        # Get stone quality prediction
        stone_quality_prediction = predict_stone_quality(img_array)
        
        # Handle case where stone quality prediction fails
        if stone_quality_prediction is None:
            logger.warning("Stone quality prediction failed, using default")
            stone_quality_prediction = {
                'quality': 'Unknown',
                'confidence': 0,
                'all_predictions': {'Good': 0, 'Bad': 0, 'Medium': 0},
                'maintenance_required': False,
                'maintenance_urgency': 'low'
            }
        
        response = {
            'status': 'success',
            'model_used': model_name,
            'model_accuracy': '96%',
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'stone_quality_assessment': stone_quality_prediction,
            'timestamp': datetime.now().isoformat(),
            'class_descriptions': get_class_descriptions(predicted_class),
            'characteristics': get_class_characteristics(predicted_class)
        }
        
        logger.info(f"Prediction successful: {predicted_class} ({confidence:.2f}%) using {model_name}")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'failed'}), 500

@app.route('/api/predict/ensemble', methods=['POST'])
def predict_ensemble():
    """
    Ensemble prediction using ONLY the 3 best performing models
    DenseNet (96%), Inception (96%), MobileNetV2 (96%)
    Returns averaged probabilities and voting result
    """
    try:
        if not request.files.get('image'):
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        img_array, original_img = preprocess_image(image_file)
        
        # AVAILABLE MODELS: DenseNet, MobileNet, Inception, MobileNetV3Small
        available_models_list = ['densenet', 'mobilenet', 'inception', 'mobilenetv3']
        
        model_predictions_dict = {}
        class_predictions = {'type-1': [], 'type-2': [], 'type-3': []}
        models_loaded = []
        
        # Get class indices
        index_to_class = {v: k for k, v in class_indices.items()}
        
        # Load and predict with available models
        for model_name in available_models_list:
            logger.info(f"Loading {model_name} for ensemble...")
            model = load_single_model(model_name)
            
            if model is None:
                logger.warning(f"Could not load {model_name}")
                continue
            
            models_loaded.append(model_name)
            
            try:
                # Get predictions from model
                predictions = model.predict(img_array, verbose=0)
                model_probs = predictions[0]  # Get first (and only) prediction
                
                logger.info(f"{model_name} predictions: {model_probs}")
                
                # Store this model's predictions
                model_predictions_dict[model_name] = {}
                
                # Map probabilities to classes
                for class_idx, class_name in index_to_class.items():
                    prob_pct = float(model_probs[class_idx] * 100)
                    model_predictions_dict[model_name][class_name] = prob_pct
                    class_predictions[class_name].append(prob_pct)
                    
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {str(e)}")
                continue
        
        # Check if we got any predictions
        if not models_loaded:
            logger.error("No models were successfully loaded for ensemble")
            return jsonify({'error': 'Could not load any models', 'status': 'failed'}), 500
        
        # Calculate average confidence for each class
        avg_confidences = {}
        for class_name in class_predictions:
            if class_predictions[class_name]:
                avg_confidences[class_name] = np.mean(class_predictions[class_name])
            else:
                avg_confidences[class_name] = 0.0
        
        # Get final prediction (highest average confidence)
        final_prediction = max(avg_confidences, key=avg_confidences.get)
        final_confidence = float(avg_confidences[final_prediction])
        
        # Build detailed breakdown
        class_breakdown = {}
        for class_name in ['type-1', 'type-2', 'type-3']:
            class_breakdown[class_name] = {
                'average_confidence': float(avg_confidences[class_name]),
                'individual_model_predictions': {}
            }
            
            # Add each model's prediction for this class
            for model_name in models_loaded:
                if model_name in model_predictions_dict and class_name in model_predictions_dict[model_name]:
                    class_breakdown[class_name]['individual_model_predictions'][model_name] = float(model_predictions_dict[model_name][class_name])
        
        # Get stone quality prediction
        stone_quality_prediction = predict_stone_quality(img_array)
        
        # Handle case where stone quality prediction fails
        if stone_quality_prediction is None:
            logger.warning("Stone quality prediction failed, using default")
            stone_quality_prediction = {
                'quality': 'Unknown',
                'confidence': 0,
                'all_predictions': {'Good': 0, 'Bad': 0, 'Medium': 0},
                'maintenance_required': False,
                'maintenance_urgency': 'low'
            }
        
        response = {
            'status': 'success',
            'method': 'ensemble',
            'models_used': models_loaded,
            'num_models': len(models_loaded),
            'final_prediction': final_prediction,
            'average_confidence': final_confidence,
            'confidence_by_class': avg_confidences,
            'detailed_breakdown': class_breakdown,
            'stone_quality_assessment': stone_quality_prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Ensemble prediction successful: {final_prediction} ({final_confidence:.2f}%)")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Ensemble prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'failed', 'type': 'ensemble_error'}), 500

# ==================== WATER QUALITY ANALYSIS ====================

def analyze_water_quality(image):
    """
    Analyze water quality from image
    Returns quality metrics: clarity, color analysis, algae detection
    """
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Color analysis
        avg_color = np.mean(img_array, axis=(0, 1))
        green_ratio = avg_color[1] / (avg_color[0] + avg_color[2] + 1)  # Green channel analysis
        blue_ratio = avg_color[2] / (avg_color[0] + avg_color[2] + 1)   # Blue channel analysis
        
        # Brightness analysis (clarity indicator)
        brightness = np.mean(img_array)
        clarity_score = min(100, (brightness / 255) * 100)
        
        # Algae likelihood (high green values)
        algae_likelihood = min(100, (green_ratio - 0.3) * 100)
        
        # Turbidity (inverse of brightness)
        turbidity_score = 100 - clarity_score
        
        # Overall quality score
        quality_score = (clarity_score * 0.4) + (100 - turbidity_score) * 0.3 + (100 - algae_likelihood) * 0.3
        
        return {
            'clarity_score': float(clarity_score),
            'turbidity_score': float(turbidity_score),
            'algae_likelihood': float(max(0, algae_likelihood)),
            'brightness': float(brightness),
            'color_analysis': {
                'red': float(avg_color[0]),
                'green': float(avg_color[1]),
                'blue': float(avg_color[2])
            },
            'overall_quality': float(quality_score),
            'quality_category': categorize_water_quality(quality_score),
            'recommendations': get_water_quality_recommendations(quality_score, algae_likelihood)
        }
    
    except Exception as e:
        logger.error(f"Water quality analysis error: {str(e)}")
        return None

def categorize_water_quality(score):
    """Categorize water quality based on score"""
    if score >= 80:
        return 'Excellent'
    elif score >= 60:
        return 'Good'
    elif score >= 40:
        return 'Fair'
    elif score >= 20:
        return 'Poor'
    else:
        return 'Critical'

def get_water_quality_recommendations(quality_score, algae_likelihood):
    """Get maintenance recommendations based on water quality"""
    recommendations = []
    
    if quality_score < 40:
        recommendations.append('Immediate cleaning and maintenance required')
        recommendations.append('Check for blockages in water inlets')
    
    if algae_likelihood > 50:
        recommendations.append('High algae presence detected - consider treatment')
        recommendations.append('Reduce sunlight exposure or increase water circulation')
    
    if quality_score < 60:
        recommendations.append('Regular cleaning schedule recommended')
        recommendations.append('Monitor water quality weekly')
    
    if not recommendations:
        recommendations.append('Maintain regular monitoring schedule')
        recommendations.append('Good maintenance practices')
    
    return recommendations

@app.route('/api/water-quality', methods=['POST'])
def water_quality():
    """
    Analyze water quality from image
    """
    try:
        if 'image' in request.files:
            image_file = request.files['image']
            _, original_img = preprocess_image(image_file)
        elif request.json and 'image' in request.json:
            _, original_img = preprocess_image(request.json['image'])
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        water_quality_data = analyze_water_quality(original_img)
        
        if water_quality_data is None:
            return jsonify({'error': 'Water quality analysis failed'}), 500
        
        response = {
            'status': 'success',
            'water_quality': water_quality_data,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Water quality API error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'failed'}), 500

# ==================== CHATBOT ====================

class SimpleChatbot:
    """Simple intent-based chatbot for Pushkarani information"""
    
    def __init__(self):
        self.knowledge_base = self.load_knowledge_base()
        self.conversation_history = []
    
    def load_knowledge_base(self):
        """Load chatbot knowledge base"""
        # Use imported knowledge base if available, otherwise fallback
        if KNOWLEDGE_BASE:
            return KNOWLEDGE_BASE
        
        return {
            'types': {
                'type-1': {
                    'name': 'Teppakulam (Float Festival Tank)',
                    'definition': 'A large, shallow temple tank designed for the Teppotsavam (float festival). It features a central pavilion (Maiya Mandapam) and spectator steps.',
                    'characteristics': [
                        'Central pavilion or island structure',
                        'Large, open water surface (typically shallow)',
                        'Wide spectator steps around perimeter',
                        'Festival-centric architecture',
                        'Dimensions: typically 200m+ in length'
                    ],
                    'example': 'Vandiyur Mariamman Teppakulam, Madurai (built 1645)',
                    'patronage': 'Nayaka Dynasty (16th-18th century)',
                    'purpose': 'Ritual spectacle and public engagement'
                },
                'type-2': {
                    'name': 'Kalyani (Geometric Stepped Tank)',
                    'definition': 'An ornate, subterranean stepped tank with complex geometric patterns. Open-air but featuring intricate pyramidal steps.',
                    'characteristics': [
                        'Complex, fractal-like step patterns',
                        'Subterranean design',
                        'Symmetrical geometry',
                        'Dark stone construction (chlorite schist)',
                        'Engineering showcase'
                    ],
                    'example': 'Stepped Tank in Hampi Royal Enclosure',
                    'patronage': 'Vijayanagara Empire (14th-17th century)',
                    'purpose': 'Royal ritual use and aesthetic expression'
                },
                'type-3': {
                    'name': 'Kunda (Simple Rectangular Tank)',
                    'definition': 'The foundational, multipurpose temple tank. A simple excavated basin with linear steps.',
                    'characteristics': [
                        'Simple, rectangular or square shape',
                        'Linear stone steps (ghat)',
                        'Functional design',
                        'Wide steps for community access',
                        'Often decorated with colors'
                    ],
                    'example': 'Haridra Nadhi, Mannargudi (23 acres - largest in India)',
                    'patronage': 'Pallava Dynasty (6th-9th century)',
                    'purpose': 'Multi-purpose: ritual, storage, groundwater recharge'
                }
            },
            'general': {
                'what_is_pushkarani': 'A Pushkarani is a sacred temple pond or tank in India. It\'s not just a water body but a complex artifact of theology, ecology, engineering, and society. The term comes from Sanskrit, literally translating to "lotus pool".',
                'purpose': 'Pushkaranis serve three main purposes: (1) Sacred purification - a place for ritual bathing; (2) Water management - rainwater harvesting and groundwater recharge; (3) Social function - a community gathering space and venue for festivals.',
                'history': 'The history of Pushkaranis spans over 2,000 years, from the Great Bath of Mohenjo-Daro (3rd millennium BCE) through the Pallava, Chola, Vijayanagara, and Nayaka dynasties.',
                'engineering': 'Ancient builders used sophisticated hydraulic systems, including aqueducts and underground channels. They used lime mortar and stone to create waterproof structures that still survive centuries later.',
                'conservation': 'Over 100,000 Pushkaranis in India are not well maintained. Many water reserves are wasted due to poor upkeep, silting, and lack of community engagement.',
                'contribute': 'You can contribute by: (1) Documenting tanks in your area; (2) Participating in restoration projects; (3) Sharing information to raise awareness; (4) Supporting conservation organizations.'
            },
            'facts': [
                'The largest temple tank in India is the Haridra Nadhi in Mannargudi, Tamil Nadu, covering 23 acres (9.3 hectares).',
                'The Rani ki Vav in Patan, Gujarat is a UNESCO World Heritage site and descends 7 levels deep.',
                'Over 100,000 Pushkaranis in India are not well maintained.',
                'Ancient builders used hydraulic lime mortar made from shell lime and clay minerals.',
                'The Hampi tanks were connected via a sophisticated network spanning miles to the Tungabhadra River.',
                'The Vandiyur Teppakulam in Madurai was created by excavating clay to make bricks for Thirumalai Nayak\'s palace.',
                'Pushkaranis were sometimes used to hide temple treasures during invasions.',
                'The Swami Pushkarini at Tirumala hosts an annual 5-day Teppotsavam festival.',
                'Vastu Shastra mandates that water bodies be located in the North, Northeast, or East of temple complexes.',
                'The term "Tirtha" refers both to a pilgrimage site and a sacred water body - they are inseparable in Hindu tradition.'
            ]
        }
    
    def get_response(self, user_message):
        """Generate chatbot response based on user message"""
        user_msg_lower = user_message.lower()
        
        # Type queries
        for type_key in ['type-1', 'type-2', 'type-3', 'teppakulam', 'kalyani', 'kunda', 'float festival', 'stepped']:
            if type_key.replace('-', '') in user_msg_lower or type_key in user_msg_lower:
                # Match with actual type keys
                matched_type = None
                if any(word in user_msg_lower for word in ['type-1', 'teppakulam', 'float', 'nayaka']):
                    matched_type = 'type-1'
                elif any(word in user_msg_lower for word in ['type-2', 'kalyani', 'stepped', 'geometric', 'hampi']):
                    matched_type = 'type-2'
                elif any(word in user_msg_lower for word in ['type-3', 'kunda', 'rectangular', 'simple', 'haridra']):
                    matched_type = 'type-3'
                
                if matched_type and matched_type in self.knowledge_base['types']:
                    type_info = self.knowledge_base['types'][matched_type]
                    return {
                        'type': 'type_info',
                        'data': type_info,
                        'message': f"Here's information about {type_info['name']}:"
                    }
        
        # General knowledge queries
        for key, value in self.knowledge_base.get('general', {}).items():
            if any(word in user_msg_lower for word in key.replace('_', ' ').split()):
                return {
                    'type': 'general_info',
                    'data': value,
                    'message': value
                }
        
        # Facts request
        if any(word in user_msg_lower for word in ['fact', 'facts', 'did you know', 'interesting', 'tell me']):
            facts_list = self.knowledge_base.get('facts', [])
            if facts_list:
                return {
                    'type': 'facts',
                    'data': np.random.choice(facts_list),
                    'message': 'Here\'s an interesting fact:'
                }
        
        # Default response
        return {
            'type': 'help',
            'message': 'I can help you learn about Pushkarani (temple ponds). Ask me about: Type 1 (Teppakulam), Type 2 (Kalyani), Type 3 (Kunda), history, engineering, conservation, or interesting facts!',
            'suggestions': ['Tell me about Type-1 tanks', 'What is a Pushkarani?', 'Conservation efforts', 'Interesting facts']
        }

# Initialize chatbot
chatbot = SimpleChatbot()

@app.route('/api/chatbot', methods=['POST'])
def chatbot_endpoint():
    """Chatbot endpoint for Pushkarani queries"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        response = chatbot.get_response(user_message)
        
        return jsonify({
            'status': 'success',
            'user_message': user_message,
            'bot_response': response,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'failed'}), 500

# ==================== UTILITY ENDPOINTS ====================

def get_class_descriptions(class_name):
    """Get description for a Pushkarani type"""
    descriptions = chatbot.knowledge_base['types']
    return descriptions.get(class_name, {})

def get_class_characteristics(class_name):
    """Get characteristics for a Pushkarani type"""
    descriptions = chatbot.knowledge_base['types']
    return descriptions.get(class_name, {}).get('characteristics', [])

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available best performing models"""
    best_models_info = {
        'densenet': {'accuracy': '96%', 'speed': 'Fast', 'parameters': '7M'},
        'mobilenet': {'accuracy': '96%', 'speed': 'Very Fast', 'parameters': '3.5M'},
        'inception': {'accuracy': '95%', 'speed': 'Medium', 'parameters': '27M'},
        'mobilenetv3': {'accuracy': '96%', 'speed': 'Very Fast', 'parameters': '2.2M'}
    }
    
    return jsonify({
        'status': 'success',
        'available_models': ['densenet', 'mobilenet', 'inception', 'mobilenetv3'],
        'model_details': best_models_info,
        'total_best_models': 4,
        'ensemble_available': True,
        'classes': list(class_indices.keys()) if class_indices else None,
        'note': 'Using top 4 performing models with 95-96% accuracy each'
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'models_loaded': len(loaded_models),
            'available_models': list(loaded_models.keys()),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        print(f"[ERROR in /api/health] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/facts', methods=['GET'])
def get_facts():
    """Get random Pushkarani facts"""
    num_facts = request.args.get('count', 5, type=int)
    facts = np.random.choice(
        chatbot.knowledge_base['facts'],
        size=min(num_facts, len(chatbot.knowledge_base['facts'])),
        replace=False
    ).tolist()
    
    return jsonify({
        'status': 'success',
        'facts': facts,
        'count': len(facts)
    }), 200

@app.route('/api/conservation', methods=['GET'])
def conservation_info():
    """Get conservation and contribution information"""
    return jsonify({
        'status': 'success',
        'critical_issue': 'Over 100,000 Pushkaranis in India are not well maintained',
        'water_loss': 'Significant water reserves are wasted due to poor upkeep, silting, and lack of community engagement',
        'how_to_contribute': [
            'Document and photograph temple ponds in your area',
            'Participate in local restoration and maintenance projects',
            'Share information and raise public awareness',
            'Support conservation-focused NGOs and government initiatives',
            'Report damaged tanks to local authorities',
            'Educate community members about importance of Pushkaranis'
        ],
        'impact': 'Conservation efforts protect sacred heritage, ensure water security, and preserve cultural traditions'
    }), 200

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found', 'status': 'error'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    import sys
    import os
    
    # Ensure stdout is utf-8 compatible
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    try:
        print("=" * 60)
        print("Pushkarani Temple Pond Classification & Analysis Backend")
        print("=" * 60)
        
        # Load models
        models_loaded = load_all_models()
        
        print(f"\n[OK] Backend initialized successfully!")
        print(f"  Models loaded: {len(loaded_models)}")
        print(f"  Classes: {list(class_indices.keys()) if class_indices else 'None'}")
        print(f"  Chatbot: Ready")
        print(f"  Water Quality Analysis: Ready")
        print("\nStarting Flask server...")
        print("=" * 60)
        
        # Start server with logging suppressed for encoding issues
        import logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        logging.getLogger('__main__').setLevel(logging.ERROR)
        logging.basicConfig(level=logging.ERROR)
        
        print(f"[INFO] Binding to 127.0.0.1:5000")
        sys.stdout.flush()
        
        # Try Waitress first, fallback to Flask if needed
        try:
            from waitress import serve
            print("[DEBUG] Using Waitress server", flush=True)
            sys.stdout.flush()
            serve(app, host='127.0.0.1', port=5000, _quiet=True)
        except Exception as waitress_error:
            print(f"[WARNING] Waitress failed: {waitress_error}")
            print("[DEBUG] Falling back to Flask development server", flush=True)
            sys.stdout.flush()
            app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False, threaded=False, processes=1)
        
    except OSError as e:
        print(f"[ERROR] Port/Socket error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
